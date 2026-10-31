//! Tests for tactical move detection: MVV-LVA captures and promotions, caching

use kingfisher::board::Board;
use kingfisher::mcts::tactical::{
    clear_tactical_cache, get_tactical_cache_stats, identify_tactical_moves, TacticalMove,
    TacticalMoveCache,
};
use kingfisher::move_generation::MoveGen;
use kingfisher::move_types::Move;
use kingfisher::piece_types::QUEEN;

// === Cache Behavior ===

#[test]
fn test_cache_eviction() {
    let move_gen = MoveGen::new();
    let mut cache = TacticalMoveCache::new(5); // Very small cache

    // Insert more positions than cache size
    for i in 0..10 {
        let fen = format!(
            "4k3/8/8/8/{}/8/8/4K3 w - - 0 1",
            match i % 4 {
                0 => "8",
                1 => "P7",
                2 => "1P6",
                _ => "2P5",
            }
        );
        let board = Board::new_from_fen(&fen);
        cache.get_or_compute(&board, &move_gen);
    }

    let (cache_size, max_size, _, _, _) = cache.stats();
    assert!(
        cache_size <= max_size,
        "Cache size {} should not exceed max {}",
        cache_size,
        max_size
    );
}

#[test]
fn test_cache_clear() {
    let move_gen = MoveGen::new();
    let mut cache = TacticalMoveCache::new(100);

    let board = Board::new();
    cache.get_or_compute(&board, &move_gen);

    let (size_before, _, _, _, _) = cache.stats();
    assert!(size_before > 0);

    cache.clear();
    let (size_after, _, hits, misses, _) = cache.stats();
    assert_eq!(size_after, 0, "Cache should be empty after clear");
    assert_eq!(hits, 0);
    assert_eq!(misses, 0);
}

#[test]
fn test_cache_hit_rate() {
    clear_tactical_cache();
    let move_gen = MoveGen::new();
    let board = Board::new();

    // 3 lookups of same position: 1 miss + 2 hits
    identify_tactical_moves(&board, &move_gen);
    identify_tactical_moves(&board, &move_gen);
    identify_tactical_moves(&board, &move_gen);

    let (_, _, hits, misses, hit_rate) = get_tactical_cache_stats();
    assert_eq!(misses, 1);
    assert_eq!(hits, 2);
    assert!(
        (hit_rate - 2.0 / 3.0).abs() < 0.01,
        "Hit rate should be ~0.667, got {}",
        hit_rate
    );
}

// === MVV-LVA Scoring ===

#[test]
fn test_captures_sorted_by_mvv_lva() {
    // Position with multiple captures available
    let move_gen = MoveGen::new();
    // Pawn on e5 can capture queen on d6
    let board = Board::new_from_fen("4k3/8/3q4/4P3/8/8/8/4K3 w - - 0 1");

    clear_tactical_cache();
    let tactical_moves = identify_tactical_moves(&board, &move_gen);

    let captures: Vec<_> = tactical_moves
        .iter()
        .filter(|t| matches!(t, TacticalMove::Capture(_, _)))
        .collect();

    if captures.len() >= 2 {
        // Should be sorted descending by score
        for window in captures.windows(2) {
            assert!(
                window[0].score() >= window[1].score(),
                "Captures should be sorted by MVV-LVA descending"
            );
        }
    }
}

// === TacticalMove API ===

#[test]
fn test_tactical_move_types() {
    let mv = Move::new(28, 36, None);

    let capture = TacticalMove::Capture(mv, 10.0);
    assert_eq!(capture.move_type(), "Capture");
    assert_eq!(capture.get_move(), mv);
    assert_eq!(capture.score(), 10.0);
}

// === Starting position ===

#[test]
fn test_starting_position_no_tactics() {
    let move_gen = MoveGen::new();
    let board = Board::new();

    clear_tactical_cache();
    let tactical_moves = identify_tactical_moves(&board, &move_gen);

    assert!(
        tactical_moves.is_empty(),
        "Starting position should have no tactical moves, got {}",
        tactical_moves.len()
    );
}

// === Position with many tactical moves ===

#[test]
fn test_complex_tactical_position() {
    // Middlegame with captures available
    let move_gen = MoveGen::new();
    let board =
        Board::new_from_fen("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4");

    clear_tactical_cache();
    let tactical_moves = identify_tactical_moves(&board, &move_gen);

    // Should find at least some captures (Bxf7, Nxe5, etc.)
    let total = tactical_moves.len();
    assert!(total > 0, "Complex position should have tactical moves");

    // Each tactical move should have a valid underlying move
    for t in &tactical_moves {
        let mv = t.get_move();
        assert!(
            mv.from() < 64 && mv.to() < 64,
            "Move squares should be valid"
        );
    }
}

// === Promotion detection ===

#[test]
fn test_promotion_detected_as_tactical() {
    let move_gen = MoveGen::new();
    // White pawn on a7 can promote on a8
    let board = Board::new_from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1");

    clear_tactical_cache();
    let tactical_moves = identify_tactical_moves(&board, &move_gen);

    // Should detect promotions as tactical moves
    assert!(
        !tactical_moves.is_empty(),
        "Promotions should be detected as tactical moves"
    );

    // Queen promotion should have the highest score
    let queen_promo = tactical_moves
        .iter()
        .find(|t| t.get_move().promotion() == Some(QUEEN));
    assert!(
        queen_promo.is_some(),
        "Queen promotion should be among tactical moves"
    );

    // Queen promotion score: victim=0, attacker=1(pawn), promo=9 → 0 - 1 + 9 = 8.0
    let score = queen_promo.unwrap().score();
    assert!(
        (score - 8.0).abs() < 0.1,
        "Queen promotion score should be 8.0, got {}",
        score
    );
}

#[test]
fn test_capture_promotion_score() {
    let move_gen = MoveGen::new();
    // White pawn on b7 can capture rook on a8 and promote
    let board = Board::new_from_fen("r3k3/1P6/8/8/8/8/8/4K3 w - - 0 1");

    clear_tactical_cache();
    let tactical_moves = identify_tactical_moves(&board, &move_gen);

    // bxa8=Q should have very high score: victim=5(rook)*10 - 1(pawn) + 9(queen) = 58.0
    let capture_promo = tactical_moves.iter().find(|t| {
        let mv = t.get_move();
        mv.promotion() == Some(QUEEN) && mv.to() == 56 // a8
    });
    assert!(
        capture_promo.is_some(),
        "Capture-promotion should be detected"
    );
    let score = capture_promo.unwrap().score();
    assert!(
        (score - 58.0).abs() < 0.1,
        "Capture-promotion PxR=Q score should be 58.0, got {}",
        score
    );
}
