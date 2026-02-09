//! Sequential Probability Ratio Test (SPRT) for model evaluation.
//!
//! Implements the trinomial GSPRT used by fishtest to decide whether a candidate
//! model is stronger than the current best. Tests after each game and stops early
//! when there is enough statistical evidence.

/// Configuration for an SPRT test.
#[derive(Debug, Clone)]
pub struct SprtConfig {
    /// Elo gain under H0 (null hypothesis, e.g. 0 = no improvement)
    pub elo0: f64,
    /// Elo gain under H1 (alternative hypothesis, e.g. 10 = meaningful improvement)
    pub elo1: f64,
    /// Type I error rate (false positive — accepting H1 when H0 is true)
    pub alpha: f64,
    /// Type II error rate (false negative — accepting H0 when H1 is true)
    pub beta: f64,
}

impl SprtConfig {
    /// Compute the SPRT decision bounds (A, B).
    ///
    /// A = ln(beta / (1 - alpha))  — lower bound (accept H0 if LLR <= A)
    /// B = ln((1 - beta) / alpha)  — upper bound (accept H1 if LLR >= B)
    pub fn bounds(&self) -> (f64, f64) {
        let a = (self.beta / (1.0 - self.alpha)).ln();
        let b = ((1.0 - self.beta) / self.alpha).ln();
        (a, b)
    }
}

/// Result of an SPRT decision check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SprtResult {
    /// Enough evidence that the candidate is stronger (LLR >= B)
    AcceptH1,
    /// Enough evidence that the candidate is NOT stronger (LLR <= A)
    AcceptH0,
    /// Not enough evidence yet
    Inconclusive,
}

impl std::fmt::Display for SprtResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SprtResult::AcceptH1 => write!(f, "H1"),
            SprtResult::AcceptH0 => write!(f, "H0"),
            SprtResult::Inconclusive => write!(f, "inconclusive"),
        }
    }
}

/// Accumulated game results for SPRT tracking.
#[derive(Debug, Clone, Default)]
pub struct SprtState {
    pub wins: u32,
    pub losses: u32,
    pub draws: u32,
}

impl SprtState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn total(&self) -> u32 {
        self.wins + self.losses + self.draws
    }

    /// Compute the log-likelihood ratio (LLR) using the trinomial GSPRT formula.
    ///
    /// Returns `None` if fewer than 2 games have been played or if variance is zero.
    ///
    /// Formula (matching fishtest):
    /// ```text
    /// s0 = 1 / (1 + 10^(-elo0/400))
    /// s1 = 1 / (1 + 10^(-elo1/400))
    /// s_obs = (W + 0.5*D) / N
    /// var = (W + D/4) / N - s_obs^2
    /// var_s = var / N
    /// LLR = (s1 - s0) * (2*s_obs - s0 - s1) / (2 * var_s)
    /// ```
    pub fn compute_llr(&self, config: &SprtConfig) -> Option<f64> {
        let n = self.total();
        if n < 2 {
            return None;
        }
        let n_f = n as f64;
        let w = self.wins as f64;
        let d = self.draws as f64;

        let s0 = 1.0 / (1.0 + f64::powf(10.0, -config.elo0 / 400.0));
        let s1 = 1.0 / (1.0 + f64::powf(10.0, -config.elo1 / 400.0));

        let s_obs = (w + 0.5 * d) / n_f;
        let var = (w + d / 4.0) / n_f - s_obs * s_obs;

        if var <= 0.0 {
            return None;
        }

        let var_s = var / n_f;
        let llr = (s1 - s0) * (2.0 * s_obs - s0 - s1) / (2.0 * var_s);
        Some(llr)
    }

    /// Check the SPRT decision given current results.
    pub fn check_decision(&self, config: &SprtConfig) -> SprtResult {
        let (lower, upper) = config.bounds();
        match self.compute_llr(config) {
            None => SprtResult::Inconclusive,
            Some(llr) => {
                if llr >= upper {
                    SprtResult::AcceptH1
                } else if llr <= lower {
                    SprtResult::AcceptH0
                } else {
                    SprtResult::Inconclusive
                }
            }
        }
    }
}
