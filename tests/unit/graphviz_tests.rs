//! Tests for MCTS tree visualization

use kingfisher::board::Board;
use kingfisher::mcts::node::{MctsNode, NodeOrigin};
use kingfisher::mcts::tactical_mcts::{tactical_mcts_search, TacticalMctsConfig};
use kingfisher::mcts::inference_server::InferenceServer;
use kingfisher::move_generation::MoveGen;
use std::sync::Arc;
use std::time::Duration;

use super::common::positions;

#[test]
fn test_export_dot_basic_structure() {
    let move_gen = MoveGen::new();
    let board = Board::new();
    let root = MctsNode::new_root(board, &move_gen);
    
    let dot = root.borrow().export_dot(2, 0);
    
    // Should have DOT preamble
    assert!(dot.contains("digraph MCTS"));
    assert!(dot.contains("node [shape=record"));
    
    // Should have legend
    assert!(dot.contains("cluster_legend"));
    assert!(dot.contains("Tier 1: Gate"));
    assert!(dot.contains("Tier 3: Neural"));
}

#[test]
fn test_export_dot_mate_position_shows_gate_color() {
    let move_gen = MoveGen::new();
    let server = InferenceServer::new_mock();
    
    // Mate in 1 position
    let board = Board::new_from_fen(positions::MATE_IN_1_WHITE);
    
    let config = TacticalMctsConfig {
        max_iterations: 20,
        time_limit: Duration::from_secs(5),
        mate_search_depth: 3,
        inference_server: Some(Arc::new(server)),
        ..Default::default()
    };

    let (best_move, _, root) = tactical_mcts_search(
        board,
        &move_gen,
        config,
    );

    let dot = root.borrow().export_dot(3, 0);

    // Should find mate
    assert!(best_move.is_some());

    // Should have red (gate) colored node
    assert!(dot.contains("firebrick1"), "Gate nodes should be firebrick1 (red)");
}

#[test]
fn test_export_dot_tactical_position_has_nodes() {
    let move_gen = MoveGen::new();
    let server = InferenceServer::new_mock();

    // Position with obvious capture
    let board = Board::new_from_fen(positions::WINNING_CAPTURE);

    let config = TacticalMctsConfig {
        max_iterations: 30,
        time_limit: Duration::from_secs(5),
        mate_search_depth: 0,
        inference_server: Some(Arc::new(server)),
        ..Default::default()
    };

    let (_, _, root) = tactical_mcts_search(
        board,
        &move_gen,
        config,
    );

    let dot = root.borrow().export_dot(3, 0);

    // Should have nodes in the tree
    let node_count = dot.matches(" [label=").count();
    assert!(node_count > 1, "Tree should have multiple nodes");
}

#[test]
fn test_node_origin_enum_colors() {
    assert_eq!(NodeOrigin::Gate.to_color(), "firebrick1");
    assert_eq!(NodeOrigin::Neural.to_color(), "lightblue");
    assert_eq!(NodeOrigin::Unknown.to_color(), "white");
}

#[test]
fn test_node_origin_labels() {
    assert_eq!(NodeOrigin::Gate.to_label(), "T1:Gate");
    assert_eq!(NodeOrigin::Neural.to_label(), "T3:NN");
}

#[test]
fn test_export_dot_respects_depth_limit() {
    let move_gen = MoveGen::new();
    let server = InferenceServer::new_mock();

    // Simple endgame position for speed in debug builds
    let board = Board::new_from_fen("4k3/4p3/8/8/8/8/4P3/4K3 w - - 0 1");

    let config = TacticalMctsConfig {
        max_iterations: 50,
        time_limit: Duration::from_secs(5),
        inference_server: Some(Arc::new(server)),
        ..Default::default()
    };
    
    let (_, _, root) = tactical_mcts_search(
        board,
        &move_gen,
        config,
    );
    
    let dot_depth_1 = root.borrow().export_dot(1, 0);
    let dot_depth_3 = root.borrow().export_dot(3, 0);
    
    // Depth 3 should have more nodes
    let count_nodes_d1 = dot_depth_1.matches(" [label=").count();
    let count_nodes_d3 = dot_depth_3.matches(" [label=").count();
    
    assert!(count_nodes_d3 >= count_nodes_d1, 
        "Deeper export should have at least as many nodes");
}

#[test]
fn test_export_dot_min_visits_filter() {
    let move_gen = MoveGen::new();
    let server = InferenceServer::new_mock();

    // Simple endgame position for speed in debug builds
    let board = Board::new_from_fen("4k3/4p3/8/8/8/8/4P3/4K3 w - - 0 1");

    let config = TacticalMctsConfig {
        max_iterations: 50,
        time_limit: Duration::from_secs(5),
        inference_server: Some(Arc::new(server)),
        ..Default::default()
    };
    
    let (_, _, root) = tactical_mcts_search(
        board,
        &move_gen,
        config,
    );
    
    let dot_no_filter = root.borrow().export_dot(4, 0);
    let dot_filtered = root.borrow().export_dot(4, 5);
    
    let count_unfiltered = dot_no_filter.matches(" [label=").count();
    let count_filtered = dot_filtered.matches(" [label=").count();
    
    assert!(count_unfiltered >= count_filtered,
        "Filtered export should have fewer or equal nodes");
}

#[test]
fn test_export_dot_no_ghost_nodes() {
    let move_gen = MoveGen::new();
    let server = InferenceServer::new_mock();

    // Simple position with tactical options — fast in debug builds
    let board = Board::new_from_fen("4k3/8/8/3q4/4N3/8/8/4K3 w - - 0 1");

    let config = TacticalMctsConfig {
        max_iterations: 30,
        time_limit: Duration::from_secs(5),
        mate_search_depth: 3,
        inference_server: Some(Arc::new(server)),
        ..Default::default()
    };

    let (_, _, root) = tactical_mcts_search(
        board,
        &move_gen,
        config,
    );

    let dot = root.borrow().export_dot(3, 0);

    // Ghost nodes were removed — no dashed/pruned nodes should appear
    assert!(!dot.contains("Pruned"), "Ghost nodes should no longer appear in DOT output");
}
