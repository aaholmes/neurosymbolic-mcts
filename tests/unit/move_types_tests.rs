//! A `Move` is packed into 16 bits: 6 for each square and 4 for the promotion.

use kingfisher::move_types::{Move, NULL_MOVE};
use kingfisher::piece_types::{BISHOP, KNIGHT, QUEEN, ROOK};
use std::collections::HashSet;

const PROMOTIONS: [Option<usize>; 5] = [None, Some(KNIGHT), Some(BISHOP), Some(ROOK), Some(QUEEN)];

fn every_move() -> impl Iterator<Item = (usize, usize, Option<usize>)> {
    (0..64)
        .flat_map(|from| (0..64).flat_map(move |to| PROMOTIONS.iter().map(move |&p| (from, to, p))))
}

#[test]
fn a_move_takes_two_bytes() {
    assert_eq!(std::mem::size_of::<Move>(), 2);
    assert!(std::mem::size_of::<Option<Move>>() <= 4);
}

#[test]
fn every_square_pair_and_promotion_round_trips() {
    let mut seen = HashSet::new();
    for (from, to, promotion) in every_move() {
        let mv = Move::new(from, to, promotion);
        assert_eq!(
            (mv.from(), mv.to(), mv.promotion()),
            (from, to, promotion),
            "{from} {to} {promotion:?}"
        );
        assert_eq!(mv.is_promotion(), promotion.is_some());
        assert!(seen.insert(mv), "two moves share an encoding: {mv:?}");
    }
    assert_eq!(seen.len(), 64 * 64 * 5);
}

#[test]
fn moves_sort_by_from_then_to_then_promotion() {
    let mut moves: Vec<Move> = every_move().map(|(f, t, p)| Move::new(f, t, p)).collect();
    moves.reverse();
    moves.sort();
    let keys: Vec<_> = moves
        .iter()
        .map(|m| (m.from(), m.to(), m.promotion()))
        .collect();
    let mut expected = keys.clone();
    expected.sort();
    assert_eq!(keys, expected);
}

#[test]
fn uci_text_round_trips() {
    for (from, to, promotion) in every_move() {
        let last_rank = to / 8 == 0 || to / 8 == 7;
        if promotion.is_some() && !last_rank {
            continue;
        }
        let mv = Move::new(from, to, promotion);
        assert_eq!(Move::from_uci(&mv.to_uci()), Some(mv), "{}", mv.to_uci());
    }
}

#[test]
fn flipping_a_move_flips_both_squares_and_keeps_the_promotion() {
    let mv = Move::new(48, 57, Some(ROOK)); // a7b8=R
    let flipped = mv.flip_vertical();
    assert_eq!(
        (flipped.from(), flipped.to(), flipped.promotion()),
        (8, 1, Some(ROOK))
    );
    assert_eq!(flipped.flip_vertical(), mv);
}

#[test]
fn the_null_move_is_a1_to_a1() {
    assert_eq!(NULL_MOVE, Move::null());
    assert_eq!(
        (NULL_MOVE.from(), NULL_MOVE.to(), NULL_MOVE.promotion()),
        (0, 0, None)
    );
}
