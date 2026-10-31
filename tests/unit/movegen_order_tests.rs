//! Move generation must keep its exact output order: MCTS creates children, and
//! the network's legal-move priors are matched to moves, in generator order.
//! The fixture records `gen_pseudo_legal_moves` and `is_checkmate_or_stalemate`
//! for ~800 positions from seeded random games; regenerate it only on purpose with
//! `cargo test write_movegen_order_fixture -- --ignored`.

use kingfisher::board::Board;
use kingfisher::move_generation::{move_list_capacity, MoveGen, MAX_MOVES};
use kingfisher::move_types::Move;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

const FIXTURE: &str = "tests/fixtures/movegen_order.txt";

fn uci_list(moves: &[Move]) -> String {
    moves
        .iter()
        .map(|m| m.to_uci())
        .collect::<Vec<_>>()
        .join(" ")
}

fn record(board: &Board, move_gen: &MoveGen) -> String {
    let (captures, quiet) = move_gen.gen_pseudo_legal_moves(board);
    let (mate, stalemate) = board.is_checkmate_or_stalemate(move_gen);
    format!(
        "{}|{}|{}|{}|{}",
        board.to_fen().unwrap(),
        uci_list(&captures),
        uci_list(&quiet),
        mate,
        stalemate
    )
}

fn legal(board: &Board, move_gen: &MoveGen) -> Vec<Move> {
    let (captures, quiet) = move_gen.gen_pseudo_legal_moves(board);
    captures
        .into_iter()
        .chain(quiet)
        .filter(|m| board.apply_move_to_board(*m).is_legal(move_gen))
        .collect()
}

/// Positions from 40 seeded random games, every third ply, plus special cases.
fn sample_positions(move_gen: &MoveGen) -> Vec<Board> {
    let mut out: Vec<Board> = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1",
        "4R1k1/5ppp/8/8/8/8/8/6K1 b - - 1 1",
        "k7/1R6/K7/8/8/8/8/8 b - - 0 1",
        "8/8/8/pP6/8/8/8/K6k w - a6 0 1",
        "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/P7/8/8/8/8/8/K6k w - - 0 1",
        "3r4/2P5/8/8/8/8/k7/4K3 w - - 0 1",
    ]
    .iter()
    .map(|f| Board::new_from_fen(f))
    .collect();
    let mut rng = StdRng::seed_from_u64(20261030);
    for _ in 0..40 {
        let mut board = Board::new();
        for ply in 0..200 {
            let moves = legal(&board, move_gen);
            if moves.is_empty() {
                out.push(board.clone());
                break;
            }
            if ply % 3 == 0 {
                out.push(board.clone());
            }
            board = board.apply_move_to_board(*moves.choose(&mut rng).unwrap());
        }
    }
    out
}

#[test]
#[ignore]
fn write_movegen_order_fixture() {
    let move_gen = MoveGen::new();
    let lines: Vec<String> = sample_positions(&move_gen)
        .iter()
        .map(|b| record(b, &move_gen))
        .collect();
    std::fs::write(FIXTURE, lines.join("\n") + "\n").unwrap();
}

#[test]
fn move_generation_keeps_its_recorded_order_and_terminal_flags() {
    let move_gen = MoveGen::new();
    let text = std::fs::read_to_string(FIXTURE).expect("fixture present");
    let mut n = 0;
    for line in text.lines().filter(|l| !l.is_empty()) {
        let fen = line.split('|').next().unwrap();
        let board = Board::new_from_fen(fen);
        assert_eq!(record(&board, &move_gen), line, "position {fen}");
        n += 1;
    }
    assert!(n > 500, "fixture has only {n} positions");
}

#[test]
fn checkmate_or_stalemate_matches_a_full_legal_move_count() {
    let move_gen = MoveGen::new();
    let text = std::fs::read_to_string(FIXTURE).expect("fixture present");
    let mut terminal = 0;
    for line in text.lines().filter(|l| !l.is_empty()) {
        let board = Board::new_from_fen(line.split('|').next().unwrap());
        let none_legal = legal(&board, &move_gen).is_empty();
        let in_check = board.is_check(&move_gen);
        let expected = (none_legal && in_check, none_legal && !in_check);
        assert_eq!(board.is_checkmate_or_stalemate(&move_gen), expected);
        terminal += none_legal as usize;
    }
    assert!(terminal >= 2, "fixture should include mates and stalemates");
}

#[test]
fn move_lists_reserve_room_for_every_move_only_when_asked() {
    assert_eq!(move_list_capacity(None), 0);
    assert_eq!(move_list_capacity(Some("0")), 0);
    assert_eq!(move_list_capacity(Some("1")), MAX_MOVES);
}

#[test]
#[should_panic(expected = "KF_RESERVE_MOVES")]
fn an_unknown_move_reserve_setting_is_rejected() {
    move_list_capacity(Some("yes"));
}
