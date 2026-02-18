//! Interactive Play — play against the Caissawary engine from the terminal.
//!
//! Usage:
//!   play_engine                                              # Classical fallback (no NN)
//!   play_engine --model weights/gen_18.pt --simulations 200  # With a trained model
//!   play_engine --play-as black --koth                       # Play as black, KOTH enabled
//!   play_engine --model m.pt --disable-tier1 --disable-material  # Vanilla model
//!   play_engine --list-models runs/long_run/scaleup_2m_tiered/weights

use std::env;
use std::io::{self, Write};
use std::sync::Arc;
use std::time::Duration;

use kingfisher::boardstack::BoardStack;
use kingfisher::mcts::{tactical_mcts_search, InferenceServer, TacticalMctsConfig};
use kingfisher::move_generation::MoveGen;
use kingfisher::move_types::Move;
use kingfisher::neural_net::NeuralNetPolicy;

fn main() {
    let args: Vec<String> = env::args().collect();

    // --list-models: print .pt files in a directory and exit
    if let Some(dir) = parse_arg_str(&args, "--list-models") {
        list_models(&dir);
        return;
    }

    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_usage();
        return;
    }

    let model_path = parse_arg_str(&args, "--model");
    let simulations: u32 = parse_arg(&args, "--simulations").unwrap_or(800);
    let enable_koth = args.iter().any(|a| a == "--koth");
    let disable_tier1 = args.iter().any(|a| a == "--disable-tier1");
    let disable_material = args.iter().any(|a| a == "--disable-material");
    let play_as_black = parse_arg_str(&args, "--play-as")
        .map(|s| s.to_lowercase() == "black")
        .unwrap_or(false);

    // Human plays white by default, engine plays the other side
    let human_is_white = !play_as_black;

    println!("=== Caissawary Interactive Play ===");
    println!("Simulations/move: {}", simulations);
    println!("KOTH: {}", if enable_koth { "enabled" } else { "disabled" });
    println!(
        "Tiers: {}",
        if disable_tier1 && disable_material {
            "disabled (vanilla)"
        } else if disable_tier1 {
            "tier1 disabled"
        } else if disable_material {
            "material disabled"
        } else {
            "enabled"
        }
    );
    if let Some(ref path) = model_path {
        println!("Model: {}", path);
    } else {
        println!("Model: none (classical fallback)");
    }
    println!("You play: {}", if human_is_white { "White" } else { "Black" });
    println!();
    println!("Commands: quit, undo, fen, moves, help");
    println!();

    let move_gen = MoveGen::new();

    // Load NN if requested
    let server: Option<Arc<InferenceServer>> = if let Some(ref path) = model_path {
        let mut nn = NeuralNetPolicy::new();
        if let Err(e) = nn.load(path) {
            eprintln!("Failed to load model: {}", e);
            std::process::exit(1);
        }
        println!("Model loaded successfully.");
        Some(Arc::new(InferenceServer::new(nn, 1)))
    } else {
        None
    };

    let mut board_stack = BoardStack::new();
    let mut ply = 0u32;

    loop {
        let board = board_stack.current_state().clone();
        let side = if board.w_to_move { "White" } else { "Black" };
        let is_human_turn = board.w_to_move == human_is_white;

        // Print board
        println!();
        println!(
            "--- Move {} | {} to move ---",
            ply / 2 + 1,
            side
        );
        board.print();

        // Check termination
        let (mate, stalemate) = board.is_checkmate_or_stalemate(&move_gen);
        if mate {
            let winner = if board.w_to_move { "Black" } else { "White" };
            println!("Checkmate! {} wins.", winner);
            break;
        }
        if stalemate {
            println!("Stalemate! Draw.");
            break;
        }
        if enable_koth {
            let (w_koth, b_koth) = board.is_koth_win();
            if w_koth {
                println!("KOTH win! White reaches the center.");
                break;
            }
            if b_koth {
                println!("KOTH win! Black reaches the center.");
                break;
            }
        }
        if board_stack.is_draw_by_repetition() {
            println!("Draw by threefold repetition.");
            break;
        }
        if board.halfmove_clock() >= 100 {
            println!("Draw by 50-move rule.");
            break;
        }

        if is_human_turn {
            // Human's turn
            let (captures, quiets) = move_gen.gen_pseudo_legal_moves(&board);
            let legal_moves: Vec<Move> = captures
                .iter()
                .chain(quiets.iter())
                .filter(|m| board.is_legal_after_move(**m, &move_gen))
                .copied()
                .collect();
            loop {
                let input = prompt("Your move> ");
                let trimmed = input.trim().to_lowercase();

                match trimmed.as_str() {
                    "quit" | "exit" | "q" => {
                        println!("Goodbye!");
                        return;
                    }
                    "undo" => {
                        // Undo two plies (human + engine) if possible
                        if ply >= 2 {
                            board_stack.undo_move();
                            board_stack.undo_move();
                            ply -= 2;
                            println!("Undone 2 plies.");
                        } else if ply >= 1 {
                            board_stack.undo_move();
                            ply -= 1;
                            println!("Undone 1 ply.");
                        } else {
                            println!("Nothing to undo.");
                        }
                        break;
                    }
                    "fen" => {
                        println!(
                            "{}",
                            board.to_fen().unwrap_or_else(|| "???".to_string())
                        );
                        continue;
                    }
                    "moves" => {
                        let mut move_strs: Vec<String> =
                            legal_moves.iter().map(|m| m.to_uci()).collect();
                        move_strs.sort();
                        println!("{} legal moves: {}", move_strs.len(), move_strs.join(" "));
                        continue;
                    }
                    "help" => {
                        println!("Enter moves in UCI notation (e.g. e2e4, e7e8q)");
                        println!("Commands: quit, undo, fen, moves, help");
                        continue;
                    }
                    _ => {}
                }

                // Try to parse as a move
                let parsed = Move::from_uci(&trimmed);
                match parsed {
                    Some(mv) if legal_moves.contains(&mv) => {
                        board_stack.make_move(mv);
                        ply += 1;
                        break;
                    }
                    Some(_) => {
                        println!("Illegal move. Type 'moves' to see legal moves.");
                    }
                    None => {
                        println!("Could not parse '{}'. Use UCI notation (e.g. e2e4).", trimmed);
                    }
                }
            }
        } else {
            // Engine's turn
            println!("Thinking...");

            let config = TacticalMctsConfig {
                max_iterations: simulations,
                time_limit: Duration::from_secs(600),
                mate_search_depth: 5,
                exploration_constant: 1.414,
                use_neural_policy: server.is_some(),
                inference_server: server.clone(),
                enable_koth,
                enable_tier1_gate: !disable_tier1,
                enable_tier3_neural: server.is_some(),
                enable_material_value: !disable_material,
                ..Default::default()
            };

            let (best_move, _stats, root) =
                tactical_mcts_search(board.clone(), &move_gen, config);

            // Print thinking summary
            let root_ref = root.borrow();

            // Check if it's a forced win (mate or KOTH gate)
            if let Some(mate_mv) = root_ref.mate_move {
                if let Some(dist) = root_ref.terminal_distance {
                    println!("Forced mate in {}!", dist.div_ceil(2));
                } else {
                    println!("Forced win found!");
                }
                drop(root_ref);
                println!("Engine plays: {}", mate_mv.to_uci());
                board_stack.make_move(mate_mv);
                ply += 1;
            } else if let Some(mv) = best_move {
                // Print top candidate moves
                let mut children_info: Vec<(Move, u32, f64)> = Vec::new();
                for child in &root_ref.children {
                    let c = child.borrow();
                    if let Some(cmv) = c.action {
                        let q = if c.visits > 0 {
                            c.total_value / c.visits as f64
                        } else {
                            0.0
                        };
                        children_info.push((cmv, c.visits, q));
                    }
                }
                children_info.sort_by(|a, b| b.1.cmp(&a.1));

                let top_n = children_info.len().min(5);
                if top_n > 0 {
                    println!("{:<8} {:>6}  {:>8}", "Move", "Visits", "Q");
                    for (cmv, visits, q) in &children_info[..top_n] {
                        println!("{:<8} {:>6}  {:>+8.3}", cmv.to_uci(), visits, q);
                    }
                }

                drop(root_ref);
                println!("Engine plays: {}", mv.to_uci());
                board_stack.make_move(mv);
                ply += 1;
            } else {
                drop(root_ref);
                println!("Engine has no legal moves.");
                break;
            }
        }
    }
}

fn prompt(msg: &str) -> String {
    print!("{}", msg);
    io::stdout().flush().unwrap();
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap_or(0);
    input
}

fn list_models(dir: &str) {
    let path = std::path::Path::new(dir);
    if !path.is_dir() {
        eprintln!("'{}' is not a directory.", dir);
        std::process::exit(1);
    }

    let mut entries: Vec<String> = std::fs::read_dir(path)
        .unwrap_or_else(|e| {
            eprintln!("Cannot read '{}': {}", dir, e);
            std::process::exit(1);
        })
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(".pt") {
                Some(name)
            } else {
                None
            }
        })
        .collect();

    entries.sort();
    if entries.is_empty() {
        println!("No .pt files found in '{}'.", dir);
    } else {
        println!("Models in '{}':", dir);
        for name in &entries {
            println!("  {}", name);
        }
        println!("({} models)", entries.len());
    }
}

fn parse_arg(args: &[String], flag: &str) -> Option<u32> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}

fn parse_arg_str(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn print_usage() {
    println!("Usage: play_engine [OPTIONS]");
    println!();
    println!("Options:");
    println!("  --model <path>          Path to a TorchScript model (.pt)");
    println!("  --simulations <n>       MCTS simulations per move (default: 800)");
    println!("  --play-as <white|black> Which side to play (default: white)");
    println!("  --koth                  Enable King of the Hill variant");
    println!("  --disable-tier1         Disable Tier 1 safety gates");
    println!("  --disable-material      Disable material integration in value");
    println!("  --list-models <dir>     List .pt files in a directory and exit");
    println!("  --help, -h              Show this help");
}
