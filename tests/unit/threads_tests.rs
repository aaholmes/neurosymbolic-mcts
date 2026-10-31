//! CPU lists, thread affinity and per-thread CPU time (`kingfisher::threads`).

use kingfisher::threads::{
    current_thread_cpus, parse_cpu_list, pin_current_thread, thread_cpu_time,
};

#[test]
fn cpu_lists_parse_singles_ranges_and_mixes() {
    assert_eq!(parse_cpu_list("0"), Ok(vec![0]));
    assert_eq!(parse_cpu_list("0,16"), Ok(vec![0, 16]));
    assert_eq!(parse_cpu_list("2-4,0"), Ok(vec![0, 2, 3, 4]));
    assert_eq!(parse_cpu_list(" 1 , 1 "), Ok(vec![1]));
}

#[test]
fn malformed_cpu_lists_are_rejected() {
    for bad in ["", "a", "3-1", "1,,2", "-1", "1-"] {
        assert!(parse_cpu_list(bad).is_err(), "{bad:?} should be rejected");
    }
}

#[cfg(target_os = "linux")]
#[test]
fn pinning_a_thread_restricts_it_to_those_cpus() {
    std::thread::spawn(|| {
        let all = current_thread_cpus().unwrap();
        assert!(!all.is_empty());
        let first = vec![all[0]];
        pin_current_thread(&first).unwrap();
        assert_eq!(current_thread_cpus().unwrap(), first);
        pin_current_thread(&all).unwrap();
        assert_eq!(current_thread_cpus().unwrap(), all);
    })
    .join()
    .unwrap();
}

#[cfg(target_os = "linux")]
#[test]
fn pinning_to_no_cpus_is_an_error() {
    std::thread::spawn(|| assert!(pin_current_thread(&[]).is_err()))
        .join()
        .unwrap();
}

#[test]
fn thread_cpu_time_counts_work_but_not_sleep() {
    use std::time::{Duration, Instant};
    let t0 = thread_cpu_time();
    std::thread::sleep(Duration::from_millis(30));
    let slept = thread_cpu_time() - t0;
    assert!(
        slept < Duration::from_millis(10),
        "sleep used {slept:?} of CPU"
    );

    // Spin until this thread has used 15 ms of CPU; on a loaded machine that may
    // take longer than 15 ms of wall time, but it must happen.
    let t1 = thread_cpu_time();
    let start = Instant::now();
    let mut x = 0u64;
    while thread_cpu_time() - t1 < Duration::from_millis(15) {
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "spinning used no CPU time"
        );
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
    }
    std::hint::black_box(x);
}

#[test]
fn other_cpus_excludes_the_reserved_ones() {
    use kingfisher::threads::other_cpus;
    assert_eq!(other_cpus(&[0, 1, 2, 16, 17], &[0, 16]), vec![1, 2, 17]);
    assert_eq!(other_cpus(&[0, 1], &[]), vec![0, 1]);
    // Reserving every CPU would leave the game threads nowhere to run
    assert_eq!(other_cpus(&[0, 1], &[0, 1]), vec![0, 1]);
}

#[test]
fn named_spans_nest_and_close_without_a_profiler() {
    use kingfisher::threads::{span, PhaseClock, PhaseTiming};
    let mut phase = PhaseTiming::default();
    {
        let _outer = span(c"kf.test.outer");
        let clock = PhaseClock::start(c"kf.test.phase");
        let _inner = span(c"kf.test.inner");
        clock.stop(&mut phase);
    }
    assert_eq!(phase.wall.count, 1);
}
