//! Side positions in self-play: at some main-line moves, one or two random
//! moves are played from the position and the result is searched once, for a
//! training sample with a policy target and no value target.
use kingfisher::board::Board;
use kingfisher::move_generation::MoveGen;
use kingfisher::training_data::{
    random_line, save_binary_data, SidePositions, TrainingSample, NO_VALUE,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn counts(side: &SidePositions, n: usize) -> [usize; 3] {
    let mut rng = StdRng::seed_from_u64(1);
    let mut c = [0; 3];
    for _ in 0..n {
        c[side.random_moves(&mut rng) as usize] += 1;
    }
    c
}

#[test]
fn side_positions_are_off_by_default() {
    let side = SidePositions::default();
    assert_eq!(side.fraction, 0.0);
    assert_eq!(side.two_moves, 0.5);
    assert_eq!(counts(&side, 1000), [1000, 0, 0]);
}

#[test]
fn fraction_sets_how_often_and_two_moves_sets_how_many() {
    let one = SidePositions {
        fraction: 1.0,
        two_moves: 0.0,
    };
    assert_eq!(counts(&one, 1000), [0, 1000, 0]);
    let two = SidePositions {
        fraction: 1.0,
        two_moves: 1.0,
    };
    assert_eq!(counts(&two, 1000), [0, 0, 1000]);
    let [none, one, two] = counts(
        &SidePositions {
            fraction: 0.5,
            two_moves: 0.5,
        },
        10_000,
    );
    // Expected 5,000 / 2,500 / 2,500; each within about 4 standard deviations.
    assert!((4800..=5200).contains(&none), "{none}");
    assert!((2300..=2700).contains(&one), "{one}");
    assert!((2300..=2700).contains(&two), "{two}");
}

#[test]
fn settings_outside_zero_to_one_are_rejected() {
    assert!(SidePositions::new(1.5, 0.5).is_err());
    assert!(SidePositions::new(0.5, -0.1).is_err());
    assert!(SidePositions::new(f64::NAN, 0.5).is_err());
    assert_eq!(SidePositions::new(0.25, 0.75).unwrap().fraction, 0.25);
}

#[test]
fn a_random_line_plays_legal_moves_and_keeps_the_turn_order() {
    let mg = MoveGen::new();
    let start = Board::new();
    for seed in 0..50 {
        let mut rng = StdRng::seed_from_u64(seed);
        let one = random_line(&start, 1, &mg, &mut rng).expect("start has moves");
        assert!(!one.w_to_move);
        assert!(one.is_legal(&mg));
        let two = random_line(&start, 2, &mg, &mut rng).expect("start has moves");
        assert!(two.w_to_move);
        assert!(two.is_legal(&mg));
        assert_ne!(two.to_fen(), start.to_fen());
    }
}

#[test]
fn a_random_line_covers_many_first_moves() {
    let mg = MoveGen::new();
    let mut rng = StdRng::seed_from_u64(0);
    let distinct: std::collections::HashSet<String> = (0..400)
        .map(|_| {
            random_line(&Board::new(), 1, &mg, &mut rng)
                .unwrap()
                .to_fen()
                .unwrap()
        })
        .collect();
    assert_eq!(distinct.len(), 20, "uniform over the 20 opening moves");
}

#[test]
fn a_random_line_stops_where_the_game_ends() {
    let mg = MoveGen::new();
    let mut rng = StdRng::seed_from_u64(0);
    // Fool's mate: White is checkmated and has no move.
    let mated =
        Board::new_from_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3");
    assert!(random_line(&mated, 1, &mg, &mut rng).is_none());
    // Black to move can only mate or play on; a line that mates has no position to search.
    let mate_in_one = Board::new_from_fen("r5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1");
    for _ in 0..50 {
        if let Some(b) = random_line(&mate_in_one, 2, &mg, &mut rng) {
            let (mate, stalemate) = b.is_checkmate_or_stalemate(&mg);
            assert!(!mate && !stalemate);
        }
    }
}

#[test]
fn a_sample_without_a_value_target_is_saved_as_nan() {
    let dir = std::env::temp_dir().join(format!("side_pos_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("g.bin");
    let sample = |value_target| TrainingSample {
        board: Board::new(),
        policy: vec![(0, 1.0)],
        policy_moves: vec![],
        value_target,
        material_scalar: 0.0,
        qsearch_completed: true,
        w_to_move: true,
    };
    save_binary_data(path.to_str().unwrap(), &[sample(1.0), sample(NO_VALUE)]).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(floats.len(), 2 * 5763);
    assert_eq!(floats[1090], 1.0);
    assert!(floats[5763 + 1090].is_nan());
    std::fs::remove_dir_all(&dir).unwrap();
}
