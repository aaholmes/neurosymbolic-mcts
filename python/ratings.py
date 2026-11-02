"""Joint Elo ratings from match results, under the Bradley-Terry model.

Each match record is {"a", "b", "wins", "losses", "draws"}, counted from a's
side. A draw counts as half a win for each side. One engine, the anchor, is
fixed at 0 Elo. A weak normal prior (standard deviation 1,000 Elo) keeps an
engine with a perfect score at a finite rating.
"""

import numpy as np
from scipy.optimize import minimize

ELO_SCALE = np.log(10) / 400  # converts Elo differences to logits


def _players(matches):
    names = []
    for m in matches:
        for n in (m["a"], m["b"]):
            if n not in names:
                names.append(n)
    return names


def fit(matches, anchor, prior_sd=1000.0):
    """Maximum a posteriori Elo of every engine that played, with anchor = 0."""
    names = _players(matches)
    if anchor not in names:
        raise ValueError(f"the anchor {anchor!r} has played no games")
    free = [n for n in names if n != anchor]
    index = {n: i for i, n in enumerate(free)}
    a = np.array([index.get(m["a"], -1) for m in matches])
    b = np.array([index.get(m["b"], -1) for m in matches])
    score = np.array([m["wins"] + 0.5 * m["draws"] for m in matches], dtype=float)
    total = np.array([m["wins"] + m["losses"] + m["draws"] for m in matches], dtype=float)

    def neg_log_posterior(r):
        full = np.append(r, 0.0)  # index -1 is the anchor
        x = ELO_SCALE * (full[a] - full[b])
        log_p = -np.logaddexp(0.0, -x)
        log_q = -np.logaddexp(0.0, x)
        nll = -(score * log_p + (total - score) * log_q).sum() + (r ** 2).sum() / (2 * prior_sd ** 2)
        resid = ELO_SCALE * (score - total * np.exp(log_p))  # d log-likelihood / d (r_a - r_b)
        grad = np.zeros(len(full))
        np.add.at(grad, a, -resid)
        np.add.at(grad, b, resid)
        return nll, grad[:-1] + r / prior_sd ** 2

    if not free:
        return {anchor: 0.0}
    result = minimize(neg_log_posterior, np.zeros(len(free)), jac=True, method="L-BFGS-B")
    out = {n: float(result.x[index[n]]) for n in free}
    out[anchor] = 0.0
    return out


def bootstrap_se(matches, anchor, n_boot=200, rng=None):
    """Standard error of each rating, resampling the games within each match."""
    rng = rng or np.random.default_rng(0)
    draws = []
    for _ in range(n_boot):
        resampled = []
        for m in matches:
            n = m["wins"] + m["losses"] + m["draws"]
            if n == 0:
                continue
            w, l, d = rng.multinomial(n, np.array([m["wins"], m["losses"], m["draws"]]) / n)
            resampled.append(dict(m, wins=int(w), losses=int(l), draws=int(d)))
        draws.append(fit(resampled, anchor))
    names = _players(matches)
    return {n: float(np.std([r[n] for r in draws], ddof=1)) for n in names}


def choose_opponents(ratings, anchor, k):
    """The anchor plus k - 1 engines spread evenly across the rating range,
    or every engine when there are no more than k."""
    others = sorted((n for n in ratings if n != anchor), key=lambda n: ratings[n])
    if len(others) < k:
        return [anchor] + others
    picks = np.unique(np.round(np.linspace(0, len(others) - 1, k - 1)).astype(int))
    return [anchor] + [others[i] for i in picks]
