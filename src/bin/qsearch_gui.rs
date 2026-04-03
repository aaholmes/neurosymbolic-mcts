//! Interactive GUI for visualizing quiescence search trees.
//!
//! Run with: cargo run --bin qsearch_gui
//! Then open http://localhost:8088 in your browser.

use kingfisher::board::Board;
use kingfisher::boardstack::BoardStack;
use kingfisher::eval::PestoEval;
use kingfisher::move_generation::MoveGen;
use kingfisher::move_types::Move;
use kingfisher::piece_types::{BLACK, BISHOP, KING, KNIGHT, PAWN, QUEEN, ROOK, WHITE};
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
    eval_cp: i32,
    score_cp: i32,
    move_uci: Option<String>,
    move_san: Option<String>,
    is_capture: bool,
    is_check: bool,
    is_evasion: bool,
    is_fork: bool,
    children: Vec<QSearchTreeNode>,
}

#[derive(Serialize)]
struct QSearchResponse {
    tree: QSearchTreeNode,
    nodes_searched: u32,
    extended: bool,
}

// ── Tactical quiet detection (mirrors quiescence.rs logic) ─────────────────

fn compute_fork_targets(board: &Board, mv: Move, move_gen: &MoveGen) -> u64 {
    let piece_type = match board.get_piece(mv.from) {
        Some((_, pt)) => pt,
        None => return 0,
    };
    let enemy_color = if board.w_to_move { BLACK } else { WHITE };

    match piece_type {
        PAWN => {
            let attack_bb = if board.w_to_move {
                move_gen.wp_capture_bitboard[mv.to]
            } else {
                move_gen.bp_capture_bitboard[mv.to]
            };
            let enemy_valuable = board.get_piece_bitboard(enemy_color, KNIGHT)
                | board.get_piece_bitboard(enemy_color, BISHOP)
                | board.get_piece_bitboard(enemy_color, ROOK)
                | board.get_piece_bitboard(enemy_color, QUEEN)
                | board.get_piece_bitboard(enemy_color, KING);
            let forked = attack_bb & enemy_valuable;
            if forked.count_ones() >= 2 { forked } else { 0 }
        }
        KNIGHT => {
            let attack_bb = move_gen.n_move_bitboard[mv.to];
            let enemy_high_value = board.get_piece_bitboard(enemy_color, ROOK)
                | board.get_piece_bitboard(enemy_color, QUEEN)
                | board.get_piece_bitboard(enemy_color, KING);
            let forked = attack_bb & enemy_high_value;
            if forked.count_ones() >= 2 { forked } else { 0 }
        }
        _ => 0,
    }
}

fn is_tactical_quiet(board: &Board, mv: Move, move_gen: &MoveGen) -> bool {
    if board.gives_check(mv, move_gen) {
        return true;
    }
    compute_fork_targets(board, mv, move_gen) != 0
}

// ── Basic Q-search tree builder (captures only) ────────────────────────────

fn build_qsearch_tree(
    board: &mut BoardStack,
    move_gen: &MoveGen,
    pesto: &PestoEval,
    depth: u8,
    nodes: &mut u32,
) -> QSearchTreeNode {
    *nodes += 1;
    let fen = board.current_state().to_fen().unwrap_or_default();
    let stand_pat = pesto.pst_eval_cp(board.current_state());

    let leaf = |eval, score| QSearchTreeNode {
        fen: fen.clone(), eval_cp: eval, score_cp: score,
        move_uci: None, move_san: None,
        is_capture: false, is_check: false, is_evasion: false, is_fork: false,
        children: vec![],
    };

    if depth == 0 {
        return leaf(stand_pat, stand_pat);
    }

    let captures = move_gen.gen_pseudo_legal_captures(board.current_state());
    let mut scored_children: Vec<(Move, i32, QSearchTreeNode)> = Vec::new();

    for capture in &captures {
        board.make_move(*capture);
        if !board.current_state().is_legal(move_gen) {
            board.undo_move();
            continue;
        }
        let child_tree = build_qsearch_tree(board, move_gen, pesto, depth - 1, nodes);
        let score = -child_tree.score_cp;
        board.undo_move();
        scored_children.push((*capture, score, child_tree));
    }

    scored_children.sort_by(|a, b| b.1.cmp(&a.1));
    scored_children.truncate(TOP_N);

    let mut best_score = stand_pat;
    let mut children = Vec::new();

    for (mv, score, mut child_tree) in scored_children {
        if score > best_score { best_score = score; }
        child_tree.move_uci = Some(mv.to_uci());
        child_tree.move_san = Some(move_to_san(board.current_state(), &mv, move_gen));
        child_tree.is_capture = true;
        children.push(child_tree);
    }

    QSearchTreeNode {
        fen, eval_cp: stand_pat, score_cp: best_score,
        move_uci: None, move_san: None,
        is_capture: false, is_check: false, is_evasion: false, is_fork: false,
        children,
    }
}

// ── Extended Q-search tree builder (captures + checks + forks + evasions) ──

fn build_ext_qsearch_tree(
    board: &mut BoardStack,
    move_gen: &MoveGen,
    pesto: &PestoEval,
    depth: u8,
    white_tactic_used: bool,
    black_tactic_used: bool,
    forked_pieces: u64,
    nodes: &mut u32,
) -> QSearchTreeNode {
    *nodes += 1;
    let fen = board.current_state().to_fen().unwrap_or_default();
    let in_check = board.current_state().is_check(move_gen);

    let leaf = |eval, score| QSearchTreeNode {
        fen: fen.clone(), eval_cp: eval, score_cp: score,
        move_uci: None, move_san: None,
        is_capture: false, is_check: false, is_evasion: false, is_fork: false,
        children: vec![],
    };

    let stand_pat = pesto.pst_eval_cp(board.current_state());

    if depth == 0 {
        return leaf(stand_pat, stand_pat);
    }

    let stm_is_white = board.current_state().w_to_move;
    let stm_tactic_used = if stm_is_white { white_tactic_used } else { black_tactic_used };

    // Collect scored children: (move, parent-perspective score, tree, is_capture, is_check, is_evasion, is_fork)
    let mut scored_children: Vec<(Move, i32, QSearchTreeNode, bool, bool, bool, bool)> = Vec::new();

    if in_check {
        // In check: all legal moves are evasions
        let (caps, quiets) = move_gen.gen_pseudo_legal_moves(board.current_state());
        let mut any_legal = false;
        for mv in caps.iter().chain(quiets.iter()) {
            let is_cap = board.current_state().get_piece(mv.to).is_some()
                || (board.current_state().get_piece(mv.from).map_or(false, |(_, pt)| pt == PAWN)
                    && mv.from % 8 != mv.to % 8);
            board.make_move(*mv);
            if !board.current_state().is_legal(move_gen) {
                board.undo_move();
                continue;
            }
            any_legal = true;
            let child_tree = build_ext_qsearch_tree(
                board, move_gen, pesto, depth - 1,
                white_tactic_used, black_tactic_used, 0, nodes,
            );
            let score = -child_tree.score_cp;
            board.undo_move();
            scored_children.push((*mv, score, child_tree, is_cap, false, true, false));
        }
        if !any_legal {
            // Checkmate
            return leaf(stand_pat, -1_000_000);
        }
    } else {
        // Not in check: captures first
        let captures = move_gen.gen_pseudo_legal_captures(board.current_state());
        for capture in &captures {
            board.make_move(*capture);
            if !board.current_state().is_legal(move_gen) {
                board.undo_move();
                continue;
            }
            let child_tree = build_ext_qsearch_tree(
                board, move_gen, pesto, depth - 1,
                white_tactic_used, black_tactic_used, 0, nodes,
            );
            let score = -child_tree.score_cp;
            board.undo_move();
            scored_children.push((*capture, score, child_tree, true, false, false, false));
        }

        // Tactical quiets + forked piece retreats
        if !stm_tactic_used || forked_pieces != 0 {
            let (_, quiets) = move_gen.gen_pseudo_legal_moves(board.current_state());
            for mv in &quiets {
                let from_bit = 1u64 << mv.from;
                let is_forked_retreat = forked_pieces & from_bit != 0;
                let is_tactical = !stm_tactic_used
                    && is_tactical_quiet(board.current_state(), *mv, move_gen);

                if !is_tactical && !is_forked_retreat {
                    continue;
                }

                let new_forked = if is_tactical {
                    compute_fork_targets(board.current_state(), *mv, move_gen)
                } else {
                    0
                };

                board.make_move(*mv);
                if !board.current_state().is_legal(move_gen) {
                    board.undo_move();
                    continue;
                }

                let new_w_used = if is_tactical && stm_is_white { true } else { white_tactic_used };
                let new_b_used = if is_tactical && !stm_is_white { true } else { black_tactic_used };

                let gives_check = board.current_state().is_check(move_gen);
                let is_fork_move = new_forked != 0;

                let child_tree = build_ext_qsearch_tree(
                    board, move_gen, pesto, depth - 1,
                    new_w_used, new_b_used, new_forked, nodes,
                );
                let score = -child_tree.score_cp;
                board.undo_move();
                scored_children.push((*mv, score, child_tree, false, gives_check, is_forked_retreat, is_fork_move));
            }
        }
    }

    scored_children.sort_by(|a, b| b.1.cmp(&a.1));
    scored_children.truncate(TOP_N);

    let mut best_score = if in_check { -1_000_000 } else { stand_pat };
    let mut children = Vec::new();

    for (mv, score, mut child_tree, is_cap, is_chk, is_evasion, is_fork) in scored_children {
        if score > best_score { best_score = score; }
        child_tree.move_uci = Some(mv.to_uci());
        child_tree.move_san = Some(move_to_san(board.current_state(), &mv, move_gen));
        child_tree.is_capture = is_cap;
        child_tree.is_check = is_chk;
        child_tree.is_evasion = is_evasion;
        child_tree.is_fork = is_fork;
        children.push(child_tree);
    }

    QSearchTreeNode {
        fen, eval_cp: stand_pat, score_cp: best_score,
        move_uci: None, move_san: None,
        is_capture: false, is_check: false, is_evasion: false, is_fork: false,
        children,
    }
}

// ── SAN formatting ─────────────────────────────────────────────────────────

fn piece_char(piece_type: usize) -> &'static str {
    match piece_type {
        0 => "",
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
        || (piece_type == 0 && mv.from % 8 != mv.to % 8);

    let dest = sq_to_algebraic(mv.to);

    let mut san = if piece_type == 0 {
        if is_capture {
            let from_file = (b'a' + (mv.from % 8) as u8) as char;
            format!("{}x{}", from_file, dest)
        } else {
            dest.clone()
        }
    } else if piece_type == KING {
        // Castling detection
        if mv.from == 4 && mv.to == 6 { return "O-O".to_string(); }
        if mv.from == 4 && mv.to == 2 { return "O-O-O".to_string(); }
        if mv.from == 60 && mv.to == 62 { return "O-O".to_string(); }
        if mv.from == 60 && mv.to == 58 { return "O-O-O".to_string(); }
        let pc = piece_char(piece_type);
        if is_capture { format!("{}x{}", pc, dest) } else { format!("{}{}", pc, dest) }
    } else {
        let pc = piece_char(piece_type);
        if is_capture { format!("{}x{}", pc, dest) } else { format!("{}{}", pc, dest) }
    };

    if let Some(promo) = mv.promotion {
        san.push('=');
        san.push_str(match promo {
            1 => "N", 2 => "B", 3 => "R", 4 => "Q", _ => "?",
        });
    }

    let new_board = board.apply_move_to_board(*mv);
    if new_board.is_check(move_gen) {
        let (checkmate, _) = new_board.is_checkmate_or_stalemate(move_gen);
        if checkmate { san.push('#'); } else { san.push('+'); }
    }

    san
}

// ── HTTP server ────────────────────────────────────────────────────────────

fn parse_query(query: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for pair in query.split('&') {
        if let Some((k, v)) = pair.split_once('=') {
            map.insert(k.to_string(), urldecode(v));
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
                1 => "n".to_string(), 2 => "b".to_string(),
                3 => "r".to_string(), 4 => "q".to_string(),
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

fn handle_qsearch(fen: &str, extended: bool, move_gen: &MoveGen, pesto: &PestoEval) -> String {
    let mut board = BoardStack::new_from_fen(fen);
    let mut nodes = 0u32;
    let tree = if extended {
        build_ext_qsearch_tree(
            &mut board, move_gen, pesto,
            MAX_DEPTH, false, false, 0, &mut nodes,
        )
    } else {
        build_qsearch_tree(
            &mut board, move_gen, pesto,
            MAX_DEPTH, &mut nodes,
        )
    };
    let resp = QSearchResponse { tree, nodes_searched: nodes, extended };
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
                    &b"Content-Type"[..], &b"text/html; charset=utf-8"[..],
                ).unwrap();
                let _ = request.respond(Response::from_string(html).with_header(header));
            }
            "/api/state" => {
                let fen = query.get("fen").map(|s| s.as_str())
                    .unwrap_or("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
                let uci_move = query.get("move").map(|s| s.as_str());
                let json = handle_state(fen, uci_move, &move_gen, &pesto);
                let header = Header::from_bytes(
                    &b"Content-Type"[..], &b"application/json"[..],
                ).unwrap();
                let _ = request.respond(Response::from_string(json).with_header(header));
            }
            "/api/qsearch" => {
                let fen = query.get("fen").map(|s| s.as_str())
                    .unwrap_or("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
                let extended = query.get("extended").map(|s| s == "1").unwrap_or(false);
                let json = handle_qsearch(fen, extended, &move_gen, &pesto);
                let header = Header::from_bytes(
                    &b"Content-Type"[..], &b"application/json"[..],
                ).unwrap();
                let _ = request.respond(Response::from_string(json).with_header(header));
            }
            _ => {
                let _ = request.respond(Response::from_string("Not Found").with_status_code(404));
            }
        }
    }
}
