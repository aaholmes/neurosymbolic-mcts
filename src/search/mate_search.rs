use crate::boardstack::BoardStack;
use crate::move_generation::MoveGen;
use crate::move_types::Move;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Result of a mate search
#[derive(Clone, Debug)]
struct MateResult {
    score: i32,      // 1000000 for mate, -1000000 for mated
    best_move: Move,
    depth: i32,      // Ply depth where mate was found
}

/// Shared context for all parallel searches
struct SearchContext {
    nodes_remaining: AtomicUsize,
    stop_signal: AtomicBool,
    best_result: Mutex<Option<MateResult>>,
}

impl SearchContext {
    fn new(node_budget: usize) -> Self {
        Self {
            nodes_remaining: AtomicUsize::new(node_budget),
            stop_signal: AtomicBool::new(false),
            best_result: Mutex::new(None),
        }
    }

    fn should_stop(&self) -> bool {
        self.stop_signal.load(Ordering::Relaxed) || self.nodes_remaining.load(Ordering::Relaxed) == 0
    }

    fn decrement_nodes(&self) {
        if self.nodes_remaining.load(Ordering::Relaxed) > 0 {
             self.nodes_remaining.fetch_sub(1, Ordering::Relaxed);
        }
    }
    
    fn report_mate(&self, result: MateResult) {
        let mut best = self.best_result.lock().unwrap();
        
        let is_improvement = match &*best {
            None => true,
            Some(old) => {
                if result.score >= 1_000_000 && old.score >= 1_000_000 {
                    result.depth < old.depth
                } else {
                    result.score > old.score
                }
            }
        };

        if is_improvement {
            *best = Some(result);
            if best.as_ref().unwrap().score >= 1_000_000 {
                self.stop_signal.store(true, Ordering::Relaxed);
            }
        }
    }
}

/// Public API: Parallel Mate Search
pub fn mate_search(
    board: &mut BoardStack,
    move_gen: &MoveGen,
    max_depth: i32,
    _verbose: bool, // Verbose flag ignored for parallel search simplicity
) -> (i32, Move, i32) {
    // Total node budget shared across all threads
    let total_budget = 1_000_000; 
    let context = Arc::new(SearchContext::new(total_budget));

    // Clone board for each thread
    let mut board_a = board.clone();
    let mut board_b = board.clone();
    let mut board_c = board.clone();

    // Run the portfolio in parallel
    rayon::scope(|s| {
        // Strategy A: "Spearhead" (Checks Only) - Deep, narrow
        s.spawn(|_| {
            iterative_deepening_wrapper(
                &context, 
                &mut board_a, 
                move_gen, 
                max_depth * 2, // Go deeper because it's narrower
                MateStrategy::ChecksOnly
            );
        });

        // Strategy B: "Flanker" (One Quiet Move) - Medium depth
        s.spawn(|_| {
            iterative_deepening_wrapper(
                &context, 
                &mut board_b, 
                move_gen, 
                (max_depth + 2).min(6), 
                MateStrategy::OneQuiet
            );
        });

        // Strategy C: "Guardsman" (Exhaustive) - Shallow, complete
        s.spawn(|_| {
            iterative_deepening_wrapper(
                &context, 
                &mut board_c, 
                move_gen, 
                max_depth.min(4), 
                MateStrategy::Exhaustive
            );
        });
    });

    // Retrieve and validate result
    let final_res = context.best_result.lock().unwrap().clone();
    let nodes_searched = total_budget - context.nodes_remaining.load(Ordering::Relaxed);
    
    if let Some(res) = final_res {
        // Double check legality in the ACTUAL root position
        let (captures, quiet) = move_gen.gen_pseudo_legal_moves(&board.current_state());
        let is_pseudo_legal = captures.iter().any(|m| *m == res.best_move) || quiet.iter().any(|m| *m == res.best_move);
        if is_pseudo_legal {
            board.make_move(res.best_move);
            let is_legal = board.current_state().is_legal(move_gen);
            board.undo_move();
            if is_legal {
                return (res.score, res.best_move, nodes_searched as i32);
            }
        }
    }

    // Fallback: Return first legal move if no mate found
    let (captures, quiet) = move_gen.gen_pseudo_legal_moves(&board.current_state());
    for m in captures.into_iter().chain(quiet.into_iter()) {
        if board.current_state().get_piece(m.from).is_none() {
            continue;
        }
        board.make_move(m);
        if board.current_state().is_legal(move_gen) {
            board.undo_move();
            return (0, m, nodes_searched as i32);
        }
        board.undo_move();
    }

    (0, Move::null(), nodes_searched as i32)
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum MateStrategy {
    ChecksOnly,
    OneQuiet,
    Exhaustive,
}

fn iterative_deepening_wrapper(
    ctx: &Arc<SearchContext>,
    board: &mut BoardStack,
    move_gen: &MoveGen,
    max_depth: i32,
    strategy: MateStrategy,
) {
    for d in 1..=max_depth {
        if ctx.should_stop() { break; }
        
        let depth = 2 * d - 1; // Only check odd depths (mate for us)
        
        // Call the specific recursive search function
        let (score, best_move) = mate_search_recursive(ctx, board, move_gen, depth, -1_000_001, 1_000_001, true, strategy, 0);
        
        // If we found a forced mate at the root, report it
        if score >= 1_000_000 && best_move != Move::null() {
            // Validate legality one last time at this level
            board.make_move(best_move);
            let is_legal = board.current_state().is_legal(move_gen);
            board.undo_move();
            
            if is_legal {
                ctx.report_mate(MateResult {
                    score,
                    best_move,
                    depth: d * 2 - 1, // Store actual ply depth
                });
                break;
            }
        }
    }
}

fn mate_search_recursive(
    ctx: &SearchContext,
    board: &mut BoardStack,
    move_gen: &MoveGen,
    depth: i32,
    mut alpha: i32,
    beta: i32,
    side_to_move_is_root: bool,
    strategy: MateStrategy,
    quiet_moves_used: i32,
) -> (i32, Move) {
    ctx.decrement_nodes();
    if ctx.should_stop() { return (0, Move::null()); }

    // --- Base Case: Checkmate/Stalemate ---
    let (is_checkmate, is_stalemate) = board.current_state().is_checkmate_or_stalemate(move_gen);
    if is_checkmate {
        return (-1_000_000 - depth, Move::null());
    }
    if is_stalemate || board.is_draw_by_repetition() {
        return (0, Move::null());
    }

    // --- Base Case: Depth 0 ---
    if depth <= 0 {
        return (0, Move::null());
    }

    // --- Move Generation & Strategy Filtering ---
    let (captures, moves) = move_gen.gen_pseudo_legal_moves(&board.current_state());
    let mut legal_moves = Vec::new();

    let mut process_candidates = |candidates: Vec<Move>| {
        for m in candidates {
            if board.current_state().get_piece(m.from).is_none() {
                continue;
            }
            board.make_move(m);
            if board.current_state().is_legal(move_gen) {
                let is_check = board.current_state().is_check(move_gen);
                
                let keep = match strategy {
                    MateStrategy::ChecksOnly => is_check,
                    MateStrategy::OneQuiet => is_check || quiet_moves_used < 1,
                    MateStrategy::Exhaustive => true,
                };

                if keep {
                    legal_moves.push(m);
                }
            }
            board.undo_move();
        }
    };

    process_candidates(captures);
    process_candidates(moves);

    if legal_moves.is_empty() {
        return (0, Move::null());
    }

    let mut best_move = Move::null();
    let mut best_score = -1_000_001;

    for m in legal_moves {
        board.make_move(m);
        
        let gives_check = board.current_state().is_check(move_gen);
        let new_quiet_count = if !gives_check { quiet_moves_used + 1 } else { quiet_moves_used };
        
        let (mut score, _) = mate_search_recursive(
            ctx, board, move_gen, depth - 1, -beta, -alpha, false, strategy, new_quiet_count
        );
        
        score = -score;
        board.undo_move();
        
        if ctx.should_stop() { return (0, Move::null()); }

        if score > best_score {
            best_score = score;
            best_move = m;
        }
        
        if score > alpha {
            alpha = score;
            if side_to_move_is_root && score >= 1_000_000 && m != Move::null() {
                 ctx.report_mate(MateResult {
                    score,
                    best_move: m,
                    depth: 0, // Not used here
                });
            }
        }
        if alpha >= beta { break; }
    }

    (best_score, best_move)
}