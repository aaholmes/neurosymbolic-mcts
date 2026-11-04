//! Statistics for `qsearch_bench`, which times the extended quiescence search
//! on logged self-play leaves.

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
