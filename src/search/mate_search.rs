use crate::boardstack::BoardStack;
use crate::move_generation::MoveGen;
use crate::move_types::Move;
use rayon::prelude::*;
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
        // Decrement roughly every node, or batch it for performance if needed.
        // For simplicity, we relax strictness here; exact node counts aren't critical.
        if self.nodes_remaining.load(Ordering::Relaxed) > 0 {
             self.nodes_remaining.fetch_sub(1, Ordering::Relaxed);
        }
    }
    
    fn report_mate(&self, result: MateResult) {
        let mut best = self.best_result.lock().unwrap();
        
        // If we already have a mate, keep the one that is faster (lower depth)
        // or better for us (higher positive score).
        let is_improvement = match &*best {
            None => true,
            Some(current) => {
                // If current is winning mate, we want a faster winning mate (lower depth)
                if current.score > 0 {
                    result.score > 0 && result.depth < current.depth
                } 
                // If current is losing (negative score), we prefer anything else
                else {
                    result.score > current.score
                }
            }
        };

        if is_improvement {
            *best = Some(result);
            // If we found a winning mate for us, stop all other searches!
            if best.as_ref().unwrap().score > 0 {
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

        /* Strategy B: "Flanker" (One Quiet Move) - Medium depth
        s.spawn(|_| {
            iterative_deepening_wrapper(
                &context, 
                &mut board_b, 
                move_gen, 
                max_depth + 2, 
                MateStrategy::OneQuiet
            );
        });

        // Strategy C: "Guardsman" (Exhaustive) - Shallow, complete
        s.spawn(|_| {
            iterative_deepening_wrapper(
                &context, 
                &mut board_c, 
                move_gen, 
                max_depth, 
                MateStrategy::Exhaustive
            );
        });
        */
    });

    // Retrieve result
    let best = context.best_result.lock().unwrap();
    let nodes_searched = total_budget - context.nodes_remaining.load(Ordering::Relaxed);
    
    match &*best {
        Some(res) => (res.score, res.best_move, nodes_searched as i32),
        None => (0, Move::null(), nodes_searched as i32),
    }
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
        if score >= 1_000_000 {
            ctx.report_mate(MateResult {
                score,
                best_move,
                depth: d * 2 - 1, // Store actual ply depth
            });
            // Don't stop here; let the shared stop_signal handle termination across threads
            // (though effectively this thread is done)
            break;
        }
    }
}

// Revised recursive function that handles all strategies
fn mate_search_recursive(
    ctx: &Arc<SearchContext>,
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
    if ctx.should_stop() {
        return (0, Move::null());
    }

    // Leaf node detection
    let (is_mate, is_stalemate) = board.current_state().is_checkmate_or_stalemate(move_gen);
    if is_mate {
        // If we (the root side) just moved, we won! Score: +1M
        // If the opponent just moved, we lost. Score: -1M
        // The alpha-beta logic handles the sign flipping, so we return a "Mated" score.
        // Wait, standard convention: return score relative to side to move.
        // If it is my turn and I am mated, score is -1M.
        return (-1_000_000, Move::null());
    }
    if is_stalemate {
        return (0, Move::null());
    }
    if depth <= 0 {
        return (0, Move::null());
    }

    let (mut captures, moves) = move_gen.gen_pseudo_legal_moves(&board.current_state());
    // Merge moves based on strategy filtering
    
    let is_in_check = board.current_state().is_check(move_gen);
    
    // FILTER MOVES based on Strategy
    let mut legal_moves = Vec::with_capacity(captures.len() + moves.len());
    
    // Helper to process a list of candidate moves
    let mut process_candidates = |candidates: Vec<Move>| {
        for m in candidates {
            board.make_move(m);
            if !board.current_state().is_legal(move_gen) {
                board.undo_move();
                continue;
            }
            
            let gives_check = board.current_state().is_check(move_gen);
            let is_capture = board.current_state().get_piece(m.to).is_some();
            
            let mut allow = false;
            
            match strategy {
                MateStrategy::Exhaustive => allow = true,
                MateStrategy::ChecksOnly => {
                    // Must give check
                    if gives_check { allow = true; }
                    // Or if we are currently in check, we must respond (forced moves allowed)
                    if is_in_check { allow = true; } 
                },
                MateStrategy::OneQuiet => {
                    if gives_check || is_in_check {
                        allow = true;
                    } else if quiet_moves_used < 1 {
                        // Allow this one quiet move
                        allow = true;
                    }
                }
            }
            
            board.undo_move();
            
            if allow {
                legal_moves.push(m);
            }
        }
    };

    process_candidates(captures);
    process_candidates(moves);

    if legal_moves.is_empty() {
        // Stalemate (since we checked for mate above)
        return (0, Move::null());
    }

    let mut best_move = Move::null();
    let mut best_score = -1_000_001;

    for m in legal_moves {
        board.make_move(m);
        
        let gives_check = board.current_state().is_check(move_gen);
        
        let new_quiet_count = if !gives_check {
             quiet_moves_used + 1
        } else {
             quiet_moves_used
        };
        
        // Pass flipped alpha/beta
        let (mut score, _) = mate_search_recursive(
            ctx, board, move_gen, depth - 1, -beta, -alpha, !side_to_move_is_root, strategy, new_quiet_count
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
            // If we found a forced mate at root, report it immediately!
            if side_to_move_is_root && score >= 1_000_000 {
                // Adjust score for ply distance to prefer faster mates
                // Note: Real mate distance logic is slightly more complex, 
                // but this suffices for the portfolio selection.
                 ctx.report_mate(MateResult {
                    score,
                    best_move: m,
                    depth: 100 - depth, // heuristic: deeper remaining depth = faster mate
                });
            }
        }
        
        if alpha >= beta {
            break;
        }
    }

    (best_score, best_move)
}