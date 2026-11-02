//! Shared training data types for self-play and evaluation game data collection.

use crate::board::Board;
use crate::move_types::Move;
use crate::tensor::board_to_planes;
use rand::Rng;
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
    pub w_to_move: bool,         // Side to move when sample was taken
}

/// Serialize training samples to a binary file compatible with the Python training script.
///
/// Format per sample: 1088 board floats + 1 material float + 1 qsearch_completed float
/// + 1 value float + 4672 policy floats = 5763 floats * 4 bytes = 23052 bytes per sample.
///
/// The file is written under a temporary name and renamed, so a trainer
/// reading `*.bin` files while self-play runs never sees half a game.
pub fn save_binary_data(filename: &str, samples: &[TrainingSample]) -> std::io::Result<()> {
    let tmp = format!("{filename}.tmp");
    let mut file = File::create(&tmp)?;
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
    drop(file);
    std::fs::rename(&tmp, filename)
}

/// The file for one self-play game; `version` names the published weights
/// that played it, in continuous training.
pub fn game_filename(dir: &str, timestamp: u64, game: usize, version: Option<u64>) -> String {
    match version {
        Some(v) => format!("{dir}/game_{timestamp}_{game}_v{v}.bin"),
        None => format!("{dir}/game_{timestamp}_{game}.bin"),
    }
}

/// The policy training target from the root's visit counts, as move
/// probabilities. One visit is removed from each move: the search gives every
/// root move (and every capture, check and fork) a first visit, and a move
/// whose only visit showed it was bad should get no weight. If no move has a
/// second visit, the raw counts are used. Moves with no weight are left out.
pub fn policy_target(root_policy: &[(Move, u32)]) -> Vec<(Move, f32)> {
    let pruned: u32 = root_policy.iter().map(|(_, v)| v.saturating_sub(1)).sum();
    let (counts, total): (Vec<(Move, u32)>, u32) = if pruned > 0 {
        let c = root_policy
            .iter()
            .map(|(m, v)| (*m, v.saturating_sub(1)))
            .collect();
        (c, pruned)
    } else {
        (
            root_policy.to_vec(),
            root_policy.iter().map(|(_, v)| *v).sum(),
        )
    };
    counts
        .into_iter()
        .filter(|(_, v)| *v > 0)
        .map(|(m, v)| (m, v as f32 / total as f32))
        .collect()
}

/// The temperature for choosing the move played at full move `move_number`
/// (1 for the first): `tau0 * decay^(move_number - 1)`, never below `tau_min`.
pub fn temperature(move_number: u32, tau0: f64, decay: f64, tau_min: f64) -> f64 {
    (tau0 * decay.powi(move_number.saturating_sub(1) as i32)).max(tau_min)
}

/// The move to play: sampled in proportion to `(visits - 1)^(1/tau)`, the
/// counts the policy target uses, so a move whose only visit was the forced
/// first one is not played. `tau = 0`, or no move with a second visit, plays
/// the most-visited move.
pub fn sample_with_temperature(
    root_policy: &[(Move, u32)],
    tau: f64,
    rng: &mut impl Rng,
) -> Option<Move> {
    let most_visited = root_policy.iter().max_by_key(|(_, v)| *v).map(|(m, _)| *m);
    let top = root_policy.iter().map(|(_, v)| v.saturating_sub(1)).max()?;
    if tau <= 0.0 || top == 0 {
        return most_visited;
    }
    // Weights relative to the largest, in log space, so small tau cannot overflow.
    let log_top = (top as f64).ln();
    let weights: Vec<f64> = root_policy
        .iter()
        .map(|(_, v)| match v.saturating_sub(1) {
            0 => 0.0,
            c => (((c as f64).ln() - log_top) / tau).exp(),
        })
        .collect();
    let mut x = rng.gen::<f64>() * weights.iter().sum::<f64>();
    for ((m, _), w) in root_policy.iter().zip(&weights) {
        if *w > 0.0 {
            if x < *w {
                return Some(*m);
            }
            x -= w;
        }
    }
    most_visited
}

/// The exploration constant in the search's PUCT rule, unless set by a flag.
pub const DEFAULT_C_PUCT: f64 = 1.414;

/// The positive number after `flag` in `args`, or [`DEFAULT_C_PUCT`] if the flag
/// is absent; an error if its value is missing, not a number, or not positive.
pub fn c_puct_arg(args: &[String], flag: &str) -> Result<f64, String> {
    let Some(i) = args.iter().position(|a| a == flag) else {
        return Ok(DEFAULT_C_PUCT);
    };
    let value = args
        .get(i + 1)
        .ok_or_else(|| format!("{flag} needs a value"))?;
    match value.parse::<f64>() {
        Ok(c) if c > 0.0 && c.is_finite() => Ok(c),
        _ => Err(format!("{flag} needs a positive number, not {value:?}")),
    }
}
