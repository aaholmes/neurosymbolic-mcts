//! Shared training data types for self-play and evaluation game data collection.

use crate::board::Board;
use crate::move_types::Move;
use crate::tensor::board_to_planes;
use std::fs::File;
use std::io::Write;

/// A single training sample: board state, MCTS policy, game outcome, and material balance.
pub struct TrainingSample {
    pub board: Board,
    pub policy: Vec<(u16, f32)>,        // (Move Index, Probability)
    pub policy_moves: Vec<(Move, f32)>, // (Move, Probability) for display
    pub value_target: f32,              // +1 (Win), -1 (Loss), 0 (Draw)
    pub material_scalar: f32,
    pub qsearch_completed: bool, // True if Q-search resolved all captures within depth limit
    pub w_to_move: bool, // Side to move when sample was taken
}

/// Serialize training samples to a binary file compatible with the Python training script.
///
/// Format per sample: 1088 board floats + 1 material float + 1 qsearch_completed float
/// + 1 value float + 4672 policy floats = 5763 floats * 4 bytes = 23052 bytes per sample.
pub fn save_binary_data(filename: &str, samples: &[TrainingSample]) -> std::io::Result<()> {
    let mut file = File::create(filename)?;
    let mut buffer = Vec::new();

    for sample in samples {
        // 1. Board Features [17, 8, 8] -> 1088 floats
        let board_planes = board_to_planes(&sample.board);

        for val in board_planes {
            buffer.extend_from_slice(&val.to_le_bytes());
        }

        // 2. Material Scalar [1 float]
        buffer.extend_from_slice(&sample.material_scalar.to_le_bytes());

        // 3. Q-search Completed Flag [1 float] (0.0 or 1.0)
        let qsearch_flag: f32 = if sample.qsearch_completed { 1.0 } else { 0.0 };
        buffer.extend_from_slice(&qsearch_flag.to_le_bytes());

        // 4. Value Target [1 float]
        buffer.extend_from_slice(&sample.value_target.to_le_bytes());

        // 5. Policy Target [4672 floats]
        let mut policy_vec = vec![0.0f32; 4672];
        for (idx, prob) in &sample.policy {
            if (*idx as usize) < 4672 {
                policy_vec[*idx as usize] = *prob;
            }
        }
        for p in policy_vec {
            buffer.extend_from_slice(&p.to_le_bytes());
        }
    }

    file.write_all(&buffer)?;
    Ok(())
}
