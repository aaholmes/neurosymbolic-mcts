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
    is_null: bool,
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
        is_capture: false, is_check: false, is_evasion: false, is_fork: false, is_null: false,
        children: vec![],
    };

    if depth == 0 {
        return leaf(stand_pat, stand_pat);
    }

    let captures = move_gen.gen_pseudo_legal_captures(board.current_state());
    let mut scored_children: Vec<(Move, i32, QSearchTreeNode)> = Vec::new();
    let mut legal_cap_count = 0;

    for capture in &captures {
        board.make_move(*capture);
        if !board.current_state().is_legal(move_gen) {
            board.undo_move();
            continue;
        }
        legal_cap_count += 1;
        if legal_cap_count > 3 {
            board.undo_move();
            break;
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
        is_capture: false, is_check: false, is_evasion: false, is_fork: false, is_null: false,
        children,
    }
}

// ── Extended Q-search tree builder (captures + checks + forks + evasions) ──

/// Mirrors `ext_pesto_qsearch_counted` exactly — same alpha-beta pruning, same
/// move ordering, same cutoffs — but captures the tree structure for visualization.
/// Every node the engine visits becomes a tree node; alpha-beta cutoffs stop
/// exploration just like the engine does.
fn build_ext_qsearch_tree(
    board: &mut BoardStack,
    move_gen: &MoveGen,
    pesto: &PestoEval,
    mut alpha: i32,
    beta: i32,
    max_depth: u8,
    white_tactic_used: bool,
    black_tactic_used: bool,
    white_null_used: bool,
    black_null_used: bool,
    nodes: &mut u32,
) -> QSearchTreeNode {
    *nodes += 1;
    let fen = board.current_state().to_fen().unwrap_or_default();
    let in_check = board.current_state().is_check(move_gen);

    let leaf = |eval, score| QSearchTreeNode {
        fen: fen.clone(), eval_cp: eval, score_cp: score,
        move_uci: None, move_san: None,
        is_capture: false, is_check: false, is_evasion: false, is_fork: false, is_null: false,
        children: vec![],
    };

    let stand_pat = pesto.pst_eval_cp(board.current_state());
    let mut children = Vec::new();

    // ── Stand-pat with null-move "deny first choice" probe (not in check) ──
    if !in_check {
        let stm_is_white = board.current_state().w_to_move;
        let stm_null_used = if stm_is_white { white_null_used } else { black_null_used };

        let adjusted_stand_pat = if !stm_null_used && max_depth > 1 {
            board.make_null_move();
            let new_w_null = true;
            let new_b_null = true;

            // Evaluate each opponent capture/tactical individually
            let opp_captures = move_gen.gen_pseudo_legal_captures(board.current_state());
            let mut scored_opp: Vec<(i32, QSearchTreeNode)> = Vec::new();
            let mut legal_cap_count = 0;

            for cap in &opp_captures {
                board.make_move(*cap);
                if !board.current_state().is_legal(move_gen) {
                    board.undo_move();
                    continue;
                }
                let child = build_ext_qsearch_tree(
                    board, move_gen, pesto,
                    -beta, -alpha, max_depth - 2,
                    white_tactic_used, black_tactic_used,
                    new_w_null, new_b_null, nodes,
                );
                let passer_score = child.score_cp; // from passer's POV
                board.undo_move();

                let mut tree_node = child;
                tree_node.move_uci = Some(cap.to_uci());
                tree_node.move_san = Some(move_to_san(board.current_state(), cap, move_gen));
                tree_node.is_capture = true;
                scored_opp.push((passer_score, tree_node));
                legal_cap_count += 1;
                if legal_cap_count >= 2 { break; } // only need top 2 (MVV-LVA sorted)
            }

            // Also evaluate one opponent tactical quiet (check or fork)
            let opp_tactic_used = if stm_is_white { black_tactic_used } else { white_tactic_used };
            if !opp_tactic_used {
                let (_, quiets) = move_gen.gen_pseudo_legal_moves(board.current_state());
                for mv in &quiets {
                    if !is_tactical_quiet(board.current_state(), *mv, move_gen) {
                        continue;
                    }
                    let is_fork_move = compute_fork_targets(board.current_state(), *mv, move_gen) != 0;
                    board.make_move(*mv);
                    if !board.current_state().is_legal(move_gen) {
                        board.undo_move();
                        continue;
                    }
                    let opp_w_used = if !stm_is_white { true } else { white_tactic_used };
                    let opp_b_used = if stm_is_white { true } else { black_tactic_used };
                    let gives_check = board.current_state().is_check(move_gen);
                    let child = build_ext_qsearch_tree(
                        board, move_gen, pesto,
                        -beta, -alpha, max_depth - 2,
                        opp_w_used, opp_b_used,
                        new_w_null, new_b_null, nodes,
                    );
                    let passer_score = child.score_cp;
                    board.undo_move();

                    let mut tree_node = child;
                    tree_node.move_uci = Some(mv.to_uci());
                    tree_node.move_san = Some(move_to_san(board.current_state(), mv, move_gen));
                    tree_node.is_check = gives_check;
                    tree_node.is_fork = is_fork_move;
                    scored_opp.push((passer_score, tree_node));
                    break; // only need one tactical quiet
                }
            }

            board.undo_null_move();

            // Sort by passer_score ascending (worst for passer first = opponent's best)
            scored_opp.sort_by_key(|(s, _)| *s);

            // Separate captures (eligible for recapture) from tactical quiets
            let mut capture_entries: Vec<(i32, usize)> = Vec::new(); // (score, index in scored_opp)
            let mut all_scores: Vec<i32> = Vec::new();
            for (i, &(score, ref node)) in scored_opp.iter().enumerate() {
                all_scores.push(score);
                // Check if this was a capture (piece exists on target square in current board)
                if let Some(uci) = &node.move_uci {
                    if let Some(mv) = Move::from_uci(uci) {
                        if board.current_state().get_piece(mv.to).is_some() {
                            capture_entries.push((score, i));
                        }
                    }
                }
            }
            all_scores.sort_unstable();
            capture_entries.sort_by_key(|(s, _)| *s);

            // Build the (pass) node with opponent's responses as children
            let mut recapture_node: Option<QSearchTreeNode> = None;
            let mut retreat_label: Option<String> = None; // e.g., "B retreats"
            let null_threat = if all_scores.len() >= 2 {
                let second_score = all_scores[1];

                // Mystery-square recapture if 2+ captures
                if capture_entries.len() >= 2 {
                    let best_cap_idx = capture_entries[0].1;
                    let second_cap_idx = capture_entries[1].1;
                    let second_cap_score = capture_entries[1].0;

                    let best_victim_sq = Move::from_uci(
                        scored_opp[best_cap_idx].1.move_uci.as_ref().unwrap()
                    ).unwrap().to;
                    let second_cap_mv = Move::from_uci(
                        scored_opp[second_cap_idx].1.move_uci.as_ref().unwrap()
                    ).unwrap();

                    // Identify the saved piece for the retreat label
                    let saved_piece_info = board.current_state().get_piece(best_victim_sq);
                    let saved_sq_name = sq_to_algebraic(best_victim_sq);
                    let saved_piece_name = saved_piece_info
                        .map(|(_, pt)| {
                            let p = piece_char(pt);
                            if p.is_empty() { "P".to_string() } else { p.to_string() }
                        })
                        .unwrap_or("?".to_string());
                    retreat_label = Some(format!("{}{} retreats", saved_piece_name, saved_sq_name));

                    // Replay: null-move, second capture, mystery recapture
                    board.make_null_move();
                    board.make_move(second_cap_mv);
                    let recapture_threat = if board.current_state().is_legal(move_gen) {
                        let recapture = Move::new(best_victim_sq, second_cap_mv.to, None);
                        let recaptured_board = board.current_state().apply_move_to_board(recapture);
                        let recapture_eval = pesto.pst_eval_cp(&recaptured_board);
                        let recapture_score = -recapture_eval;

                        // Build a visible node for the recapture
                        let dest = sq_to_algebraic(second_cap_mv.to);
                        let recapture_san = format!("{}x{}", saved_piece_name, dest);
                        let recapture_fen = recaptured_board.to_fen().unwrap_or_default();
                        recapture_node = Some(QSearchTreeNode {
                            fen: recapture_fen,
                            eval_cp: recapture_eval,
                            score_cp: recapture_score,
                            move_uci: Some(recapture.to_uci()),
                            move_san: Some(recapture_san),
                            is_capture: true, is_check: false, is_evasion: true,
                            is_fork: false, is_null: false,
                            children: vec![],
                        });

                        second_cap_score.max(recapture_score)
                    } else {
                        second_cap_score
                    };
                    board.undo_move();
                    board.undo_null_move();
                    second_score.max(recapture_threat)
                } else {
                    second_score
                }
            } else if all_scores.len() == 1 {
                stand_pat // single threat — pass fully saves
            } else {
                stand_pat
            };

            // Add null-move node: labeled as piece retreat if recapture computed, else "(pass)"
            let mut null_children: Vec<QSearchTreeNode> = scored_opp.into_iter().map(|(_, n)| n).collect();
            if let Some(rn) = recapture_node {
                null_children.push(rn);
            }
            let null_label = retreat_label.unwrap_or_else(|| "(pass)".to_string());
            let null_node = QSearchTreeNode {
                fen: fen.clone(),
                eval_cp: stand_pat,
                score_cp: null_threat,
                move_uci: None,
                move_san: Some(null_label),
                is_capture: false, is_check: false, is_evasion: false, is_fork: false,
                is_null: true,
                children: null_children,
            };
            children.push(null_node);

            stand_pat.min(null_threat)
        } else {
            stand_pat
        };

        if adjusted_stand_pat >= beta {
            return QSearchTreeNode {
                fen, eval_cp: stand_pat, score_cp: beta,
                move_uci: None, move_san: None,
                is_capture: false, is_check: false, is_evasion: false, is_fork: false, is_null: false,
                children,
            };
        }
        if adjusted_stand_pat > alpha {
            alpha = adjusted_stand_pat;
        }
    }

    if max_depth == 0 {
        return leaf(stand_pat, alpha);
    }

    let stm_is_white = board.current_state().w_to_move;
    let stm_tactic_used = if stm_is_white { white_tactic_used } else { black_tactic_used };

    if in_check {
        // ── In check: all legal moves as evasions ──
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
            let mut child_tree = build_ext_qsearch_tree(
                board, move_gen, pesto, -beta, -alpha, max_depth - 1,
                white_tactic_used, black_tactic_used,
                white_null_used, black_null_used, nodes,
            );
            let score = -child_tree.score_cp;
            board.undo_move();

            child_tree.move_uci = Some(mv.to_uci());
            child_tree.move_san = Some(move_to_san(board.current_state(), mv, move_gen));
            child_tree.is_capture = is_cap;
            child_tree.is_evasion = true;
            children.push(child_tree);

            if score >= beta {
                return QSearchTreeNode {
                    fen, eval_cp: stand_pat, score_cp: beta,
                    move_uci: None, move_san: None,
                    is_capture: false, is_check: false, is_evasion: false, is_fork: false, is_null: false,
                    children,
                };
            }
            if score > alpha { alpha = score; }
        }
        if !any_legal {
            return leaf(stand_pat, -1_000_000);
        }
    } else {
        // ── Not in check: captures, then tactical quiets ──

        // 1. Captures (MVV-LVA sorted, capped at top 3 per node)
        let captures = move_gen.gen_pseudo_legal_captures(board.current_state());
        let mut main_cap_count = 0;
        for capture in &captures {
            board.make_move(*capture);
            if !board.current_state().is_legal(move_gen) {
                board.undo_move();
                continue;
            }
            main_cap_count += 1;
            if main_cap_count > 3 {
                board.undo_move();
                break;
            }
            let mut child_tree = build_ext_qsearch_tree(
                board, move_gen, pesto, -beta, -alpha, max_depth - 1,
                white_tactic_used, black_tactic_used,
                white_null_used, black_null_used, nodes,
            );
            let score = -child_tree.score_cp;
            board.undo_move();

            child_tree.move_uci = Some(capture.to_uci());
            child_tree.move_san = Some(move_to_san(board.current_state(), capture, move_gen));
            child_tree.is_capture = true;
            children.push(child_tree);

            if score >= beta {
                return QSearchTreeNode {
                    fen, eval_cp: stand_pat, score_cp: beta,
                    move_uci: None, move_san: None,
                    is_capture: false, is_check: false, is_evasion: false, is_fork: false, is_null: false,
                    children,
                };
            }
            if score > alpha { alpha = score; }
        }

        // 2. Tactical quiets (checks, forks)
        if !stm_tactic_used {
            let (_, quiets) = move_gen.gen_pseudo_legal_moves(board.current_state());
            for mv in &quiets {
                if !is_tactical_quiet(board.current_state(), *mv, move_gen) {
                    continue;
                }
                let is_fork_move = compute_fork_targets(board.current_state(), *mv, move_gen) != 0;

                board.make_move(*mv);
                if !board.current_state().is_legal(move_gen) {
                    board.undo_move();
                    continue;
                }

                let new_w_used = if stm_is_white { true } else { white_tactic_used };
                let new_b_used = if !stm_is_white { true } else { black_tactic_used };
                let gives_check = board.current_state().is_check(move_gen);

                let mut child_tree = build_ext_qsearch_tree(
                    board, move_gen, pesto, -beta, -alpha, max_depth - 1,
                    new_w_used, new_b_used,
                    white_null_used, black_null_used, nodes,
                );
                let score = -child_tree.score_cp;
                board.undo_move();

                child_tree.move_uci = Some(mv.to_uci());
                child_tree.move_san = Some(move_to_san(board.current_state(), mv, move_gen));
                child_tree.is_check = gives_check;
                child_tree.is_fork = is_fork_move;
                children.push(child_tree);

                if score >= beta {
                    return QSearchTreeNode {
                        fen, eval_cp: stand_pat, score_cp: beta,
                        move_uci: None, move_san: None,
                        is_capture: false, is_check: false, is_evasion: false, is_fork: false, is_null: false,
                        children,
                    };
                }
                if score > alpha { alpha = score; }
            }
        }

    }

    QSearchTreeNode {
        fen, eval_cp: stand_pat, score_cp: alpha,
        move_uci: None, move_san: None,
        is_capture: false, is_check: false, is_evasion: false, is_fork: false, is_null: false,
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
            -100_000, 100_000, 20,
            false, false, false, false, &mut nodes,
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
