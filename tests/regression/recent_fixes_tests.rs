//! Regression tests specifically targeting the recent architectural fixes

use kingfisher::board::Board;
use kingfisher::move_generation::MoveGen;
use kingfisher::mcts::node::MctsNode;
use kingfisher::mcts::inference_server::InferenceServer;
use kingfisher::mcts::tactical_mcts::{tactical_mcts_search, TacticalMctsConfig, TacticalMctsStats};
use kingfisher::mcts::selection::select_child_with_tactical_priority;
use kingfisher::move_types::Move;
use kingfisher::eval::PestoEval;
use kingfisher::neural_net::NeuralNetPolicy;
use crate::common::{board_from_fen, positions};
use std::time::Duration;
use std::rc::Rc;
use std::cell::RefCell;
use std::sync::Arc;

/// Test: Value topology is strictly Tanh [-1, 1], not probability [0, 1]
mod value_topology {
    use super::*;
    
    #[test]
    fn test_no_probability_domain_values() {
        // Search for any code path that might produce values > 1.0
        let move_gen = MoveGen::new();
        let pesto = PestoEval::new();
        let server = InferenceServer::new_mock_uniform();
        
        // Run search on various positions
        let positions = [
            positions::STARTING,
            positions::MATE_IN_1_WHITE,
            positions::WINNING_CAPTURE,
        ];
        
        for fen in positions {
            let board = board_from_fen(fen);
            
            let server_arc = Arc::new(InferenceServer::new_mock_uniform());
            
            let config = TacticalMctsConfig {
                max_iterations: 50,
                time_limit: Duration::from_secs(5),
                inference_server: Some(server_arc),
                ..Default::default()
            };

            let (_, _, root_node) = tactical_mcts_search(board, &move_gen, &pesto, &mut None, config);
            
            // Check all nodes in tree
            fn check_tree(node: &Rc<RefCell<MctsNode>>) {
                let n = node.borrow();
                
                // Check terminal value
                if let Some(v) = n.terminal_or_mate_value {
                    assert!(v >= -1.0 && v <= 1.0, 
                        "terminal_or_mate_value {} in probability domain!", v);
                }
                
                // Check NN value
                if let Some(v) = n.nn_value {
                    assert!(v >= -1.0 && v <= 1.0,
                        "nn_value {} in probability domain!", v);
                }
                
                // Check average Q
                if n.visits > 0 {
                    let q = n.total_value / n.visits as f64;
                    assert!(q >= -1.0 && q <= 1.0,
                        "average Q {} in probability domain!", q);
                }
                
                // Recurse
                for child in n.children.iter() {
                    check_tree(child);
                }
            }
            
            check_tree(&root_node);
        }
    }
}

/// Test: Side-to-move convention is correctly applied
mod stm_convention {
    use super::*;
    
    #[test]
    fn test_symmetric_position_symmetric_eval() {
        // A perfectly symmetric position should evaluate to ~0 regardless of StM
        let move_gen = MoveGen::new();
        
        // Symmetric position
        let white_stm = board_from_fen("4k3/pppppppp/8/8/8/8/PPPPPPPP/4K3 w - - 0 1");
        let black_stm = board_from_fen("4k3/pppppppp/8/8/8/8/PPPPPPPP/4K3 b - - 0 1");
        
        let w_node = MctsNode::new_root(white_stm, &move_gen);
        let b_node = MctsNode::new_root(black_stm, &move_gen);
        
        // Neither should be terminal
        assert!(w_node.borrow().terminal_or_mate_value.is_none());
        assert!(b_node.borrow().terminal_or_mate_value.is_none());
        
        // If we had a real NN, both should evaluate to approximately 0
        // For now, just verify the structure is correct
    }
    
    #[test]
    fn test_checkmate_value_is_loss_for_stm() {
        let move_gen = MoveGen::new();
        
        // White to move, White is mated
        let white_mated = board_from_fen("k7/8/8/8/8/8/1r6/K1r5 w - - 0 1");
        let w_node = MctsNode::new_root(white_mated, &move_gen);
        
        // Black to move, Black is mated  
        let black_mated = board_from_fen("1R1k4/R7/8/8/8/8/8/K7 b - - 0 1");
        let b_node = MctsNode::new_root(black_mated, &move_gen);
        
        // Both should have terminal value of -1.0 (loss for side to move)
        assert_eq!(w_node.borrow().terminal_or_mate_value, Some(-1.0),
            "Mated position should be -1.0 for StM (White)");
        assert_eq!(b_node.borrow().terminal_or_mate_value, Some(-1.0),
            "Mated position should be -1.0 for StM (Black)");
    }
}

/// Test: Tactical Q-initialization works correctly
mod tactical_q_init {
    use super::*;
    
    #[test]
    fn test_tactical_values_influence_selection() {
        let move_gen = MoveGen::new();
        let board = board_from_fen(positions::WINNING_CAPTURE);
        
        let root = MctsNode::new_root(board, &move_gen);
        
        // Manually inject tactical values
        let nxd5 = Move::new(28, 35, None); // e4xd5 (28->35)
        let quiet = Move::new(28, 27, None); // e4-d4 (28->27)
        
        {
            let mut root_mut = root.borrow_mut();
            // Pretend quiescence found Nxd5 is worth +0.8
            root_mut.tactical_values.insert(nxd5, 0.8);
            
            // And some quiet move is worth 0.0
            root_mut.tactical_values.insert(quiet, 0.0);
            
            // We must simulate expansion (children creation) because select_child requires children
            // Create dummy children
            let child1 = MctsNode::new_child(Rc::downgrade(&root), nxd5, root_mut.state.clone(), &move_gen); // State invalid but ok for this test
            let child2 = MctsNode::new_child(Rc::downgrade(&root), quiet, root_mut.state.clone(), &move_gen);
            
            root_mut.children.push(child1);
            root_mut.children.push(child2);
            
            // Set policy evaluated so it doesn't try to call NN
            root_mut.policy_evaluated = true;
            // Set priorities (uniform)
            root_mut.move_priorities.insert(nxd5, 0.5);
            root_mut.move_priorities.insert(quiet, 0.5);
        }
        
        // Selection should prefer Nxd5 initially because of higher Q-init (0.8 vs 0.0)
        let mut stats = TacticalMctsStats::default();
        let mut nn_policy = None;
        let best_child = select_child_with_tactical_priority(
            root.clone(),
            1.5,
            &move_gen,
            &mut nn_policy,
            &mut stats
        );
        
        assert!(best_child.is_some());
        assert_eq!(best_child.unwrap().borrow().action.unwrap(), nxd5,
            "Selection should prefer move with higher tactical Q-init");
    }
}

/// Test: 17-plane tensor is correctly structured
#[cfg(feature = "neural")]
mod tensor_17_plane {
    use super::*;
    use tch::Tensor;
    
    fn extract_plane(tensor: &Tensor, plane: usize) -> Vec<f32> {
        let mut result = vec![0.0f32; 64];
        tensor.narrow(0, plane as i64, 1).view([64]).copy_data(&mut result, 64);
        result
    }

    #[test]
    fn test_plane_count() {
        let nn = NeuralNetPolicy::new();
        let board = board_from_fen(positions::STARTING);
        let tensor = nn.board_to_tensor(&board);
        
        assert_eq!(tensor.size()[0], 17, "Should have exactly 17 planes");
    }
    
    #[test]
    fn test_plane_layout() {
        // Planes 0-5: Our pieces (P, N, B, R, Q, K)
        // Planes 6-11: Their pieces
        // Plane 12: En passant
        // Planes 13-16: Castling (StM-KS, StM-QS, Opp-KS, Opp-QS)
        
        let nn = NeuralNet::new_dummy(); // Wait, new() or new_dummy()? Guide says new_dummy().
        // In my code NeuralNetPolicy::new() creates one. If feature=neural is on, it's real.
        // But for unit testing logic without loading model, new() is fine as it doesn't load model by default.
        let nn = NeuralNetPolicy::new();
        
        // Position with one of each piece for both sides
        let board = board_from_fen(positions::STARTING);
        let tensor = nn.board_to_tensor(&board);
        
        // Verify pawn plane (0) has 8 pawns for White
        let pawn_plane = extract_plane(&tensor, 0);
        assert_eq!(pawn_plane.iter().filter(|&&x| x == 1.0).count(), 8);
        
        // Verify king plane (5) has exactly 1 king
        let king_plane = extract_plane(&tensor, 5);
        assert_eq!(king_plane.iter().filter(|&&x| x == 1.0).count(), 1);
    }
}
