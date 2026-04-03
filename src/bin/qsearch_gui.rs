//! Interactive GUI for visualizing quiescence search trees.
//!
//! Run with: cargo run --bin qsearch_gui
//! Then open http://localhost:8088 in your browser.

use kingfisher::board::Board;
use kingfisher::boardstack::BoardStack;
use kingfisher::eval::PestoEval;
use kingfisher::move_generation::MoveGen;
use kingfisher::move_types::Move;
use serde::Serialize;
use std::collections::HashMap;
use tiny_http::{Header, Response, Server};

const TOP_N: usize = 3;
const MAX_DEPTH: u8 = 8;

// ── API types ──────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct BoardState {
    fen: String,
    white_to_move: bool,
    legal_moves: Vec<LegalMoveInfo>,
    eval_cp: i32,
}

#[derive(Serialize)]
struct LegalMoveInfo {
    uci: String,
    from: usize,
    to: usize,
    promotion: Option<String>,
    is_capture: bool,
}

#[derive(Serialize, Clone)]
struct QSearchTreeNode {
    fen: String,
    eval_cp: i32,           // static PeSTO eval at this node (STM perspective)
    score_cp: i32,          // q-search backed-up score (STM perspective)
    move_uci: Option<String>, // move that led here (None for root)
    move_san: Option<String>,
    is_capture: bool,
    children: Vec<QSearchTreeNode>,
}

#[derive(Serialize)]
struct QSearchResponse {
    tree: QSearchTreeNode,
    nodes_searched: u32,
}

// ── Q-search tree builder ──────────────────────────────────────────────────

fn build_qsearch_tree(
    board: &mut BoardStack,
    move_gen: &MoveGen,
    pesto: &PestoEval,
    alpha_in: i32,
    beta_in: i32,
    depth: u8,
    nodes: &mut u32,
) -> QSearchTreeNode {
    *nodes += 1;
    let fen = board.current_state().to_fen().unwrap_or_default();
    let stand_pat = pesto.pst_eval_cp(board.current_state());

    if depth == 0 {
        return QSearchTreeNode {
            fen,
            eval_cp: stand_pat,
            score_cp: stand_pat,
            move_uci: None,
            move_san: None,
            is_capture: false,
            children: vec![],
        };
    }

    let mut alpha = alpha_in;
    let beta = beta_in;

    if stand_pat >= beta {
        return QSearchTreeNode {
            fen,
            eval_cp: stand_pat,
            score_cp: beta,
            move_uci: None,
            move_san: None,
            is_capture: false,
            children: vec![],
        };
    }
    if stand_pat > alpha {
        alpha = stand_pat;
    }

    let captures = move_gen.gen_pseudo_legal_captures(board.current_state());

    // Collect all legal captures with their scores
    let mut scored_children: Vec<(Move, i32, QSearchTreeNode)> = Vec::new();

    for capture in &captures {
        board.make_move(*capture);
        if !board.current_state().is_legal(move_gen) {
            board.undo_move();
            continue;
        }

        let child_tree = build_qsearch_tree(
            board, move_gen, pesto,
            -beta, -alpha,
            depth - 1, nodes,
        );
        let score = -child_tree.score_cp;
        board.undo_move();

        scored_children.push((*capture, score, child_tree));
    }

    // Sort by score descending (best moves first)
    scored_children.sort_by(|a, b| b.1.cmp(&a.1));

    // Keep top N
    scored_children.truncate(TOP_N);

    let mut best_score = stand_pat;
    let mut children = Vec::new();

    for (mv, score, mut child_tree) in scored_children {
        if score > best_score {
            best_score = score;
        }

        let uci = mv.to_uci();
        let san = move_to_san(board.current_state(), &mv, move_gen);

        child_tree.move_uci = Some(uci);
        child_tree.move_san = Some(san);
        child_tree.is_capture = true;
        children.push(child_tree);
    }

    QSearchTreeNode {
        fen,
        eval_cp: stand_pat,
        score_cp: best_score.min(beta),
        move_uci: None,
        move_san: None,
        is_capture: false,
        children,
    }
}

// ── SAN formatting ─────────────────────────────────────────────────────────

fn piece_char(piece_type: usize) -> &'static str {
    match piece_type {
        0 => "",   // pawn
        1 => "N",
        2 => "B",
        3 => "R",
        4 => "Q",
        5 => "K",
        _ => "?",
    }
}

fn sq_to_algebraic(sq: usize) -> String {
    let file = (b'a' + (sq % 8) as u8) as char;
    let rank = (b'1' + (sq / 8) as u8) as char;
    format!("{}{}", file, rank)
}

fn move_to_san(board: &Board, mv: &Move, move_gen: &MoveGen) -> String {
    let (_, piece_type) = match board.get_piece(mv.from) {
        Some(p) => p,
        None => return mv.to_uci(),
    };

    let is_capture = board.get_piece(mv.to).is_some()
        || (piece_type == 0 && mv.from % 8 != mv.to % 8); // en passant

    let dest = sq_to_algebraic(mv.to);

    let mut san = if piece_type == 0 {
        // Pawn
        if is_capture {
            let from_file = (b'a' + (mv.from % 8) as u8) as char;
            format!("{}x{}", from_file, dest)
        } else {
            dest.clone()
        }
    } else {
        let pc = piece_char(piece_type);
        if is_capture {
            format!("{}x{}", pc, dest)
        } else {
            format!("{}{}", pc, dest)
        }
    };

    if let Some(promo) = mv.promotion {
        san.push('=');
        san.push_str(match promo {
            1 => "N",
            2 => "B",
            3 => "R",
            4 => "Q",
            _ => "?",
        });
    }

    // Check if move gives check
    let new_board = board.apply_move_to_board(*mv);
    if new_board.is_check(move_gen) {
        let (checkmate, _) = new_board.is_checkmate_or_stalemate(move_gen);
        if checkmate {
            san.push('#');
        } else {
            san.push('+');
        }
    }

    san
}

// ── HTTP server ────────────────────────────────────────────────────────────

fn parse_query(query: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for pair in query.split('&') {
        if let Some((k, v)) = pair.split_once('=') {
            map.insert(
                k.to_string(),
                urldecode(v),
            );
        }
    }
    map
}

fn urldecode(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '%' {
            let hex: String = chars.by_ref().take(2).collect();
            if let Ok(byte) = u8::from_str_radix(&hex, 16) {
                result.push(byte as char);
            }
        } else if c == '+' {
            result.push(' ');
        } else {
            result.push(c);
        }
    }
    result
}

fn get_legal_moves(board: &Board, move_gen: &MoveGen) -> Vec<LegalMoveInfo> {
    let (captures, quiets) = move_gen.gen_pseudo_legal_moves(board);
    let mut legal = Vec::new();
    let fen = board.to_fen().unwrap_or_default();
    let mut board_stack = BoardStack::new_from_fen(&fen);

    for mv in captures.iter().chain(quiets.iter()) {
        board_stack.make_move(*mv);
        if board_stack.current_state().is_legal(move_gen) {
            let promo = mv.promotion.map(|p| match p {
                1 => "n".to_string(),
                2 => "b".to_string(),
                3 => "r".to_string(),
                4 => "q".to_string(),
                _ => "?".to_string(),
            });
            legal.push(LegalMoveInfo {
                uci: mv.to_uci(),
                from: mv.from,
                to: mv.to,
                promotion: promo,
                is_capture: board.get_piece(mv.to).is_some()
                    || (board.get_piece(mv.from).map_or(false, |(_, pt)| pt == 0)
                        && mv.from % 8 != mv.to % 8),
            });
        }
        board_stack.undo_move();
    }
    legal
}

fn handle_state(fen: &str, uci_move: Option<&str>, move_gen: &MoveGen, pesto: &PestoEval) -> String {
    let mut board = Board::new_from_fen(fen);

    // If a move is provided, apply it
    if let Some(uci) = uci_move {
        if let Some(mv) = Move::from_uci(uci) {
            board = board.apply_move_to_board(mv);
        }
    }

    let eval_cp = pesto.pst_eval_cp(&board);
    let legal_moves = get_legal_moves(&board, move_gen);

    let state = BoardState {
        fen: board.to_fen().unwrap_or_default(),
        white_to_move: board.w_to_move,
        legal_moves,
        eval_cp,
    };
    serde_json::to_string(&state).unwrap()
}

fn handle_qsearch(fen: &str, move_gen: &MoveGen, pesto: &PestoEval) -> String {
    let mut board = BoardStack::new_from_fen(fen);
    let mut nodes = 0u32;
    let tree = build_qsearch_tree(
        &mut board, move_gen, pesto,
        -100_000, 100_000,
        MAX_DEPTH, &mut nodes,
    );
    let resp = QSearchResponse {
        tree,
        nodes_searched: nodes,
    };
    serde_json::to_string(&resp).unwrap()
}

fn main() {
    let port = std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(8088);

    let server = Server::http(format!("0.0.0.0:{}", port)).unwrap();
    println!("Q-Search GUI running at http://localhost:{}", port);

    let move_gen = MoveGen::new();
    let pesto = PestoEval::new();

    for request in server.incoming_requests() {
        let url = request.url().to_string();
        let (path, query_str) = url.split_once('?').unwrap_or((&url, ""));
        let query = parse_query(query_str);

        match path {
            "/" => {
                let html = include_str!("../../gui/index.html");
                let header = Header::from_bytes(
                    &b"Content-Type"[..],
                    &b"text/html; charset=utf-8"[..],
                ).unwrap();
                let _ = request.respond(Response::from_string(html).with_header(header));
            }
            "/api/state" => {
                let fen = query.get("fen").map(|s| s.as_str())
                    .unwrap_or("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
                let uci_move = query.get("move").map(|s| s.as_str());
                let json = handle_state(fen, uci_move, &move_gen, &pesto);
                let header = Header::from_bytes(
                    &b"Content-Type"[..],
                    &b"application/json"[..],
                ).unwrap();
                let _ = request.respond(Response::from_string(json).with_header(header));
            }
            "/api/qsearch" => {
                let fen = query.get("fen").map(|s| s.as_str())
                    .unwrap_or("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
                let json = handle_qsearch(fen, &move_gen, &pesto);
                let header = Header::from_bytes(
                    &b"Content-Type"[..],
                    &b"application/json"[..],
                ).unwrap();
                let _ = request.respond(Response::from_string(json).with_header(header));
            }
            _ => {
                let _ = request.respond(Response::from_string("Not Found").with_status_code(404));
            }
        }
    }
}
