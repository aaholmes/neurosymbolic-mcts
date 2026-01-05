//! Tests for the SearchLogger

use kingfisher::mcts::search_logger::{
    SearchLogger, Verbosity, GateReason, SelectionReason, LogSink,
};
use kingfisher::move_types::Move;

#[test]
fn test_logger_silent_produces_no_output() {
    let logger = SearchLogger::silent();
    
    logger.log_tier1_gate(
        &GateReason::MateFound { depth: 3, score: 1000000 },
        Some(Move::new(4, 60, None)),
    );
    
    // Silent logger should not panic or produce side effects
    // (We can't easily test console output, but we can verify no crash)
}

#[test]
fn test_logger_buffered_captures_output() {
    let logger = SearchLogger::buffered(Verbosity::Normal);
    
    logger.log_tier1_gate(
        &GateReason::MateFound { depth: 3, score: 1000000 },
        Some(Move::new(4, 60, None)),
    );
    
    let output = logger.get_buffer();
    assert!(output.contains("TIER 1 GATE"));
    assert!(output.contains("Mate"));
}

#[test]
fn test_logger_respects_verbosity_levels() {
    // Minimal verbosity should NOT include verbose-level messages
    let logger = SearchLogger::buffered(Verbosity::Minimal);
    
    // This is a Normal-level log
    logger.log_tier3_neural(0.5, 0.3);
    
    let output = logger.get_buffer();
    // Minimal level is less than Normal, so this should be empty
    assert!(!output.contains("TIER 3"));
    
    // But Minimal should include tier1 gates
    logger.log_tier1_gate(&GateReason::KothWin, None);
    let output = logger.get_buffer();
    assert!(output.contains("TIER 1 GATE"));
}

#[test]
fn test_gate_reason_descriptions() {
    let mate = GateReason::MateFound { depth: 3, score: 1000000 };
    assert!(mate.description().contains("Mate"));
    
    let koth = GateReason::KothWin;
    assert!(koth.description().contains("King of the Hill"));
    
    let checkmate = GateReason::Terminal { is_checkmate: true };
    assert!(checkmate.description().contains("checkmate"));
    
    let stalemate = GateReason::Terminal { is_checkmate: false };
    assert!(stalemate.description().contains("stalemate"));
}

#[test]
fn test_selection_reason_formatting() {
    let tactical = SelectionReason::TacticalPriority {
        move_type: "Capture".to_string(),
        score: 9.5,
    };
    // Just verify it doesn't panic - actual format is internal
    let _ = format!("{:?}", tactical);
    
    let ucb = SelectionReason::UcbSelection {
        q_value: 0.5,
        u_value: 0.3,
        total: 0.8,
    };
    let _ = format!("{:?}", ucb);
}

#[test]
fn test_logger_emoji_toggle() {
    let logger_emoji = SearchLogger::buffered(Verbosity::Normal)
        .with_emoji(true);
    logger_emoji.log_tier1_gate(&GateReason::KothWin, None);
    let output = logger_emoji.get_buffer();
    assert!(output.contains("🚨") || output.contains("GATE"));
    
    let logger_no_emoji = SearchLogger::buffered(Verbosity::Normal)
        .with_emoji(false);
    logger_no_emoji.log_tier1_gate(&GateReason::KothWin, None);
    let output = logger_no_emoji.get_buffer();
    assert!(output.contains("[GATE]"));
    assert!(!output.contains("🚨"));
}

#[test]
fn test_logger_search_complete() {
    let logger = SearchLogger::buffered(Verbosity::Minimal);
    
    logger.log_search_complete(
        Some(Move::new(12, 28, None)),
        500,
        1000,
        2,
    );
    
    let output = logger.get_buffer();
    assert!(output.contains("Search complete"));
    assert!(output.contains("500")); // iterations
    assert!(output.contains("1000")); // nodes
    assert!(output.contains("Mates found: 2"));
}

#[test]
fn test_iteration_summary_respects_interval() {
    let logger = SearchLogger::buffered(Verbosity::Minimal);
    
    // At Minimal, summary only logs every 100 iterations
    logger.log_iteration_summary(1, None, 10);
    assert!(logger.get_buffer().is_empty());
    
    logger.log_iteration_summary(100, None, 100);
    assert!(!logger.get_buffer().is_empty());
}