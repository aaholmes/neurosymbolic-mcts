//! The network gets two scalars for each position: the quiescence score and
//! whether that search finished. Self-play, evaluation games and leaf
//! evaluation all compute them with `qsearch_inputs`, so the values written to
//! training data are the ones the search gives the network. With the material
//! score off (vanilla), no quiescence search runs and the scalars are (0, true).

use kingfisher::board::Board;
use kingfisher::boardstack::BoardStack;
use kingfisher::eval::PestoEval;
use kingfisher::mcts::{qsearch_inputs, QSearchVariant, QsearchInputs};
use kingfisher::move_generation::MoveGen;
use kingfisher::search::{
    forced_cap1_pesto_balance, forced_ext_pesto_balance_counted, forced_principal_exchange,
};

const VARIANTS: [QSearchVariant; 3] = [
    QSearchVariant::PrincipalExchange,
    QSearchVariant::Cap1,
    QSearchVariant::Extended,
];

/// White wins a hanging queen and ends a pawn up, so the score is positive.
const HANGING_QUEEN: &str = "4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1";

/// From a logged self-play game: the extended search reaches its depth limit
/// here and reports that it did not finish.
const UNFINISHED: &str = "1r1q1bn1/2p4r/1pnkpp2/PR1p2pp/P1P1P2P/3P1N2/3Q1P2/2B1KBR1 b - h3 0 16";

fn positions() -> Vec<Board> {
    [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1",
        "k7/1R6/K7/8/8/8/8/8 b - - 0 1",
        "8/8/8/pP6/8/8/8/K6k w - a6 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "3r4/2P5/8/8/8/8/k7/4K3 w - - 0 1",
        HANGING_QUEEN,
        UNFINISHED,
    ]
    .iter()
    .map(|f| Board::new_from_fen(f))
    .collect()
}

/// What each variant's search returns, called directly.
fn direct(board: &Board, variant: QSearchVariant) -> QsearchInputs {
    let (mg, pesto) = (MoveGen::new(), PestoEval::new());
    let mut stack = BoardStack::with_board(board.clone());
    let (score, completed, nodes, depth) = match variant {
        QSearchVariant::PrincipalExchange => {
            let (s, n) = forced_principal_exchange(&mut stack, &mg, &pesto);
            (s, true, n, 0)
        }
        QSearchVariant::Cap1 => forced_cap1_pesto_balance(&mut stack, &mg, &pesto),
        QSearchVariant::Extended => forced_ext_pesto_balance_counted(&mut stack, &mg, &pesto),
    };
    QsearchInputs {
        score,
        completed,
        nodes,
        depth,
    }
}

#[test]
fn the_test_positions_include_a_won_queen_and_an_unfinished_search() {
    let queen = direct(
        &Board::new_from_fen(HANGING_QUEEN),
        QSearchVariant::Extended,
    );
    assert!(queen.score > 0.5, "{queen:?}");
    let unfinished = direct(&Board::new_from_fen(UNFINISHED), QSearchVariant::Extended);
    assert!(!unfinished.completed, "{unfinished:?}");
}

#[test]
fn with_material_on_each_variant_returns_its_own_search_result() {
    let (mg, pesto) = (MoveGen::new(), PestoEval::new());
    for board in positions() {
        for variant in VARIANTS {
            assert_eq!(
                qsearch_inputs(&board, &mg, &pesto, variant, true),
                direct(&board, variant),
                "{variant:?} {}",
                board.to_fen().unwrap()
            );
        }
    }
}

#[test]
fn with_material_off_no_search_runs_and_the_scalars_are_zero_and_finished() {
    let (mg, pesto) = (MoveGen::new(), PestoEval::new());
    let off = QsearchInputs {
        score: 0.0,
        completed: true,
        nodes: 0,
        depth: 0,
    };
    for board in positions() {
        for variant in VARIANTS {
            assert_eq!(qsearch_inputs(&board, &mg, &pesto, variant, false), off);
        }
    }
}
