//! Self-play in continuous training: each new game uses the newest weights the
//! trainer has published, and game files are written whole and name the weights.

use kingfisher::board::Board;
use kingfisher::training_data::{game_filename, save_binary_data, TrainingSample};
use kingfisher::weights_watch::{read_published, write_published, WatchedModel};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("kf_watch_{}_{}", name, std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A loader that counts its calls and returns the path it was given.
fn counting_loader(
    calls: Arc<AtomicUsize>,
) -> impl Fn(&str) -> Result<String, String> + Send + Sync {
    move |path: &str| {
        calls.fetch_add(1, Ordering::SeqCst);
        if path.contains("broken") {
            Err(format!("cannot load {path}"))
        } else {
            Ok(path.to_string())
        }
    }
}

#[test]
fn published_weights_round_trip_and_leave_no_temporary_file() {
    let dir = scratch("roundtrip");
    let pointer = dir.join("published.txt");
    write_published(&pointer, 3, "/w/cont_3.pth").unwrap();
    assert_eq!(
        read_published(&pointer),
        Some((3, "/w/cont_3.pth".to_string()))
    );
    let names: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    assert_eq!(names.len(), 1, "{names:?}");
}

#[test]
fn missing_or_malformed_pointer_reads_as_none() {
    let dir = scratch("malformed");
    assert_eq!(read_published(&dir.join("absent.txt")), None);
    let bad = dir.join("bad.txt");
    std::fs::write(&bad, "not-a-number /w/x.pth\n").unwrap();
    assert_eq!(read_published(&bad), None);
    std::fs::write(&bad, "").unwrap();
    assert_eq!(read_published(&bad), None);
}

#[test]
fn watcher_needs_published_weights_to_start() {
    let dir = scratch("start");
    let calls = Arc::new(AtomicUsize::new(0));
    assert!(WatchedModel::new(dir.join("published.txt"), counting_loader(calls)).is_err());
}

#[test]
fn watcher_loads_newer_weights_once() {
    let dir = scratch("newer");
    let pointer = dir.join("published.txt");
    write_published(&pointer, 1, "/w/cont_1.pth").unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let watched = WatchedModel::new(pointer.clone(), counting_loader(calls.clone())).unwrap();
    let (v, m) = watched.current();
    assert_eq!((v, m.as_str()), (1, "/w/cont_1.pth"));
    assert_eq!(watched.current().0, 1);
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "unchanged pointer must not reload"
    );

    write_published(&pointer, 2, "/w/cont_2.pth").unwrap();
    let (v, m) = watched.current();
    assert_eq!((v, m.as_str()), (2, "/w/cont_2.pth"));
    watched.current();
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[test]
fn watcher_ignores_versions_that_are_not_newer() {
    let dir = scratch("older");
    let pointer = dir.join("published.txt");
    write_published(&pointer, 5, "/w/cont_5.pth").unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let watched = WatchedModel::new(pointer.clone(), counting_loader(calls.clone())).unwrap();
    write_published(&pointer, 4, "/w/cont_4.pth").unwrap();
    assert_eq!(watched.current().0, 5);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn watcher_keeps_the_old_weights_when_new_ones_fail_to_load() {
    let dir = scratch("broken");
    let pointer = dir.join("published.txt");
    write_published(&pointer, 1, "/w/cont_1.pth").unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let watched = WatchedModel::new(pointer.clone(), counting_loader(calls.clone())).unwrap();
    write_published(&pointer, 2, "/w/broken_2.pth").unwrap();
    let (v, m) = watched.current();
    assert_eq!((v, m.as_str()), (1, "/w/cont_1.pth"));
    // A failed version is not retried at every game start.
    watched.current();
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[test]
fn concurrent_game_starts_load_new_weights_once() {
    let dir = scratch("concurrent");
    let pointer = dir.join("published.txt");
    write_published(&pointer, 1, "/w/cont_1.pth").unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let slow = {
        let inner = counting_loader(calls.clone());
        move |p: &str| {
            std::thread::sleep(std::time::Duration::from_millis(30));
            inner(p)
        }
    };
    let watched = Arc::new(WatchedModel::new(pointer.clone(), slow).unwrap());
    write_published(&pointer, 2, "/w/cont_2.pth").unwrap();
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let w = watched.clone();
            std::thread::spawn(move || w.current().0)
        })
        .collect();
    for h in handles {
        let v = h.join().unwrap();
        assert!(v == 1 || v == 2);
    }
    assert_eq!(watched.current().0, 2);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[test]
fn game_filenames_name_the_weights_version_when_there_is_one() {
    assert_eq!(game_filename("d", 7, 3, None), "d/game_7_3.bin");
    assert_eq!(game_filename("d", 7, 3, Some(12)), "d/game_7_3_v12.bin");
}

#[test]
fn saved_games_are_written_whole_with_no_temporary_file_left() {
    let dir = scratch("save");
    let sample = TrainingSample {
        board: Board::new(),
        policy: vec![(0, 1.0)],
        policy_moves: vec![],
        value_target: 0.0,
        material_scalar: 0.0,
        qsearch_completed: true,
        w_to_move: true,
    };
    let path = dir.join("game_1_0.bin");
    save_binary_data(path.to_str().unwrap(), &[sample]).unwrap();
    assert_eq!(std::fs::metadata(&path).unwrap().len(), 5763 * 4);
    let names: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    assert_eq!(names.len(), 1, "{names:?}");
    assert!(Path::new(&path).exists());
}
