//! Statistics for `qsearch_bench`, which times the extended quiescence search
//! on logged self-play leaves.

use crate::search::quiescence::{ExtConfig, TtScope};

/// A pruning configuration from a name: `+`-joined parts applied in order to
/// the exact search, among `default` (the default pruning), `probe1` (one
/// threat probe per side per line), `delta<cp>` (delta pruning with that
/// margin), `see` (skip losing captures), `d<plies>` (depth limit), `n<N>` (at
/// most N positions per call), `tt` (a transposition table for each call) and
/// `ttg` (one kept by each thread).
pub fn parse_config(name: &str) -> Result<ExtConfig, String> {
    let mut cfg = ExtConfig::exact();
    for part in name.split('+') {
        if part == "exact" {
        } else if part == "default" {
            cfg = ExtConfig::default();
        } else if part == "tt" {
            cfg.tt = Some(TtScope::Call);
        } else if part == "ttg" {
            cfg.tt = Some(TtScope::Thread);
        } else if let Some(n) = part.strip_prefix('n') {
            cfg.max_nodes = Some(n.parse().map_err(|_| format!("bad budget in {part:?}"))?);
        } else if part == "probe1" {
            cfg.probe_once = true;
        } else if part == "see" {
            cfg.see_prune = true;
        } else if let Some(m) = part.strip_prefix("delta") {
            cfg.delta_margin_cp = Some(m.parse().map_err(|_| format!("bad margin in {part:?}"))?);
        } else if let Some(d) = part.strip_prefix('d') {
            cfg.max_depth = d.parse().map_err(|_| format!("bad depth in {part:?}"))?;
        } else {
            return Err(format!("unknown part {part:?}"));
        }
    }
    Ok(cfg)
}

/// A checksum of every answer in order: the score's bits and the completed
/// flag. A change that keeps every answer keeps the checksum.
pub fn answer_checksum(answers: &[(f32, bool)]) -> u64 {
    answers
        .iter()
        .fold(0xcbf2_9ce4_8422_2325u64, |h, &(score, completed)| {
            let x = ((score.to_bits() as u64) << 1) | completed as u64;
            (h ^ x).wrapping_mul(0x0000_0100_0000_01b3)
        })
}

/// The nearest-rank percentile `p` (0–100] of `values`, which it sorts.
pub fn percentile(values: &mut [f64], p: f64) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    let rank = ((p / 100.0) * values.len() as f64).ceil() as usize;
    values[rank.clamp(1, values.len()) - 1]
}
