//! Move orders for the mate search: each must keep every move, put the move
//! types it names first, and leave every answer unchanged.

use kingfisher::board::Board;
use kingfisher::move_generation::MoveGen;
use kingfisher::move_types::Move;
use kingfisher::search::mate_order::{
    order_attacker_moves, order_defender_moves, AttackerOrder, DefenderOrder, MoveOrder,
};
use kingfisher::search::mate_search;

const POSITIONS: &str = include_str!("../fixtures/mate_positions.fen");

fn boards() -> Vec<Board> {
    POSITIONS
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(Board::new_from_fen)
        .collect()
}

fn checks(board: &Board, mg: &MoveGen) -> Vec<Move> {
    let mut v = Vec::new();
    mg.gen_checking_moves_into(board, &mut v);
    v
}

fn evasions(board: &Board, mg: &MoveGen) -> Vec<Move> {
    let mut v = Vec::new();
    mg.gen_evasions_into(board, &mut v);
    v
}

fn all_moves(board: &Board, mg: &MoveGen) -> Vec<Move> {
    let (captures, quiet) = mg.gen_pseudo_legal_moves(board);
    captures.into_iter().chain(quiet).collect()
}

fn sorted(mut v: Vec<Move>) -> Vec<Move> {
    v.sort_by_key(|m| (m.from(), m.to(), m.promotion()));
    v
}

fn distance(a: usize, b: usize) -> usize {
    let (fa, ra, fb, rb) = (a % 8, a / 8, b % 8, b / 8);
    fa.abs_diff(fb).max(ra.abs_diff(rb))
}

fn king_square(board: &Board, white: bool) -> usize {
    let fen = board.to_fen().unwrap();
    let placement = fen.split(' ').next().unwrap();
    let target = if white { 'K' } else { 'k' };
    let mut sq = 56usize;
    for row in placement.split('/') {
        let mut file = 0;
        for c in row.chars() {
            if let Some(n) = c.to_digit(10) {
                file += n as usize;
            } else {
                if c == target {
                    return sq + file;
                }
                file += 1;
            }
        }
        sq -= 8;
    }
    panic!("no king in {fen}");
}

#[test]
fn the_names_list_every_order_once() {
    assert_eq!(AttackerOrder::ALL.len(), 11);
    assert_eq!(DefenderOrder::ALL.len(), 7);
    assert_eq!(MoveOrder::all().len(), 77);
    assert_eq!(MoveOrder::GENERATED, MoveOrder::parse("A0,D0").unwrap());
    let a1d1 = MoveOrder::parse("A1,D1").unwrap();
    assert_eq!(a1d1.attacker, AttackerOrder::ALL[1]);
    assert_eq!(a1d1.defender, DefenderOrder::ALL[1]);
    assert_eq!(a1d1.name(), "A1,D1");
    assert!(MoveOrder::parse("A9,D0").is_some());
    assert!(MoveOrder::parse("A10,D0").is_some());
    assert!(MoveOrder::parse("A11,D0").is_none());
    assert!(MoveOrder::parse("A1").is_none());
}

#[test]
fn every_order_keeps_every_move() {
    let mg = MoveGen::new();
    for board in boards() {
        let attacker_lists = [checks(&board, &mg), all_moves(&board, &mg)];
        for order in AttackerOrder::ALL {
            for list in &attacker_lists {
                let mut v = list.clone();
                order_attacker_moves(&board, &mg, order, &mut v);
                assert_eq!(sorted(v), sorted(list.clone()), "{order:?}");
            }
        }
        let defender_list = if board.is_check(&mg) {
            evasions(&board, &mg)
        } else {
            all_moves(&board, &mg)
        };
        for order in DefenderOrder::ALL {
            let mut v = defender_list.clone();
            order_defender_moves(&board, &mg, order, &mut v);
            assert_eq!(sorted(v), sorted(defender_list.clone()), "{order:?}");
        }
    }
}

#[test]
fn generated_order_leaves_the_list_as_it_is() {
    let mg = MoveGen::new();
    for board in boards().into_iter().take(50) {
        let list = all_moves(&board, &mg);
        let mut v = list.clone();
        order_attacker_moves(&board, &mg, MoveOrder::GENERATED.attacker, &mut v);
        assert_eq!(v, list);
        order_defender_moves(&board, &mg, MoveOrder::GENERATED.defender, &mut v);
        assert_eq!(v, list);
    }
}

#[test]
fn the_requested_attacker_order_puts_a_double_check_first() {
    let mg = MoveGen::new();
    // Nd6+ and Nf6+ check with the knight and uncover the rook on e1.
    let board = Board::new_from_fen("4k3/8/8/8/4N3/8/8/4R1K1 w - - 0 1");
    let mut v = checks(&board, &mg);
    assert!(v.len() > 2);
    order_attacker_moves(
        &board,
        &mg,
        MoveOrder::parse("A1,D0").unwrap().attacker,
        &mut v,
    );
    let first = v[0].to_uci();
    assert!(first == "e4d6" || first == "e4f6", "{first}");
    assert_eq!(board.check_kinds(v[0], &mg), (true, true));
}

#[test]
fn the_requested_attacker_order_puts_a_queen_check_next_to_the_king_first() {
    let mg = MoveGen::new();
    // Qd7+ and Qd8+ are next to the king; Qa4+, Qh5+ and Qe2+ are not.
    let board = Board::new_from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1");
    let mut v = checks(&board, &mg);
    order_attacker_moves(
        &board,
        &mg,
        MoveOrder::parse("A1,D0").unwrap().attacker,
        &mut v,
    );
    let ksq = king_square(&board, false);
    let d: Vec<usize> = v.iter().map(|m| distance(m.to(), ksq)).collect();
    assert_eq!(
        d[0],
        1,
        "{:?}",
        v.iter().map(|m| m.to_uci()).collect::<Vec<_>>()
    );
    assert!(d.windows(2).all(|w| w[0] <= w[1]), "{d:?}");
}

#[test]
fn closest_first_and_farthest_first_are_opposite() {
    let mg = MoveGen::new();
    for board in boards().into_iter().take(100) {
        let ksq = king_square(&board, !board.w_to_move);
        for (order, rising) in [("A5,D0", true), ("A8,D0", false)] {
            let mut v = all_moves(&board, &mg);
            order_attacker_moves(
                &board,
                &mg,
                MoveOrder::parse(order).unwrap().attacker,
                &mut v,
            );
            let d: Vec<usize> = v.iter().map(|m| distance(m.to(), ksq)).collect();
            assert!(
                d.windows(2)
                    .all(|w| if rising { w[0] <= w[1] } else { w[0] >= w[1] }),
                "{order} {d:?}"
            );
        }
    }
}

#[test]
fn the_requested_defender_order_is_captures_then_blocks_then_king_moves() {
    let mg = MoveGen::new();
    // Nxe1 captures the checker, Be7 and Be3 block, and the king can step aside.
    let board = Board::new_from_fen("4k3/8/8/2b5/8/5n2/8/4R1K1 b - - 0 1");
    assert!(board.is_check(&mg));
    let mut v = evasions(&board, &mg);
    order_defender_moves(
        &board,
        &mg,
        MoveOrder::parse("A0,D1").unwrap().defender,
        &mut v,
    );
    let kind = |m: &Move| {
        if m.to() == 4 {
            0 // captures the rook on e1
        } else if m.from() == 60 {
            2 // king move
        } else {
            1 // block
        }
    };
    let kinds: Vec<u8> = v.iter().map(kind).collect();
    assert!(kinds.contains(&0) && kinds.contains(&1) && kinds.contains(&2));
    assert!(kinds.windows(2).all(|w| w[0] <= w[1]), "{kinds:?}");
}

#[test]
fn check_kinds_agree_with_gives_check() {
    let mg = MoveGen::new();
    for board in boards() {
        for m in all_moves(&board, &mg) {
            let (direct, uncovered) = board.check_kinds(m, &mg);
            assert_eq!(
                direct || uncovered,
                board.gives_check(m, &mg),
                "{}",
                m.to_uci()
            );
        }
    }
}

#[test]
fn every_order_gives_the_same_answers() {
    let mg = MoveGen::new();
    let boards = boards();
    // (depth, exhaustive, quiet)
    let settings = [(3, 0, 0), (2, 0, 1), (2, 2, 0)];
    let expected: Vec<Vec<bool>> = settings
        .iter()
        .map(|&(d, e, q)| {
            boards
                .iter()
                .map(|b| mate_search::has_forced_mate_quiet(b, &mg, d, e, q).0)
                .collect()
        })
        .collect();
    // Every attacker order with the requested defender order, and the reverse.
    let orders = AttackerOrder::ALL
        .iter()
        .map(|&attacker| MoveOrder {
            attacker,
            defender: DefenderOrder::ALL[1],
        })
        .chain(DefenderOrder::ALL.iter().map(|&defender| MoveOrder {
            attacker: AttackerOrder::ALL[1],
            defender,
        }));
    for order in orders {
        for (s, &(d, e, q)) in settings.iter().enumerate() {
            for (i, b) in boards.iter().enumerate() {
                let found = mate_search::has_forced_mate_ordered(b, &mg, d, e, q, order).0;
                assert_eq!(
                    found,
                    expected[s][i],
                    "{} position {i} {d},{e},{q}",
                    order.name()
                );
            }
        }
    }
}

#[test]
fn the_cheap_escape_count_puts_the_queen_check_that_leaves_fewest_escapes_first() {
    let mg = MoveGen::new();
    // Qd7+ and Qd8+ leave the king two squares; Qa4+, Qe2+ and Qh5+ leave four.
    let board = Board::new_from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1");
    let mut v = checks(&board, &mg);
    order_attacker_moves(
        &board,
        &mg,
        MoveOrder::parse("A9,D0").unwrap().attacker,
        &mut v,
    );
    let first = v[0].to_uci();
    assert!(first == "d1d7" || first == "d1d8", "{first}");
    let last = v.last().unwrap().to_uci();
    assert!(["d1a4", "d1e2", "d1h5"].contains(&last.as_str()), "{last}");
}

#[test]
fn a10_orders_checking_moves_as_a9_and_leaves_other_lists_alone() {
    let mg = MoveGen::new();
    let a9 = MoveOrder::parse("A9,D0").unwrap().attacker;
    let a10 = MoveOrder::parse("A10,D0").unwrap().attacker;
    assert!(a10.checking_moves_only() && !a9.checking_moves_only());
    for board in boards().into_iter().take(100) {
        let (mut x, mut y) = (checks(&board, &mg), checks(&board, &mg));
        order_attacker_moves(&board, &mg, a9, &mut x);
        order_attacker_moves(&board, &mg, a10, &mut y);
        assert_eq!(x, y);
    }
}

#[test]
fn the_default_order_sorts_checking_moves_by_escape_squares() {
    let mg = MoveGen::new();
    assert_eq!(MoveOrder::DEFAULT, MoveOrder::parse("A10,D0").unwrap());
    for board in boards().into_iter().take(100) {
        let default = mate_search::has_forced_mate_quiet(&board, &mg, 4, 0, 0);
        let ordered =
            mate_search::has_forced_mate_ordered(&board, &mg, 4, 0, 0, MoveOrder::DEFAULT);
        assert_eq!(default, ordered);
    }
}
