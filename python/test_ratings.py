"""Tests for joint Elo ratings from match results (Bradley-Terry model)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import ratings


def match(a, b, wins, losses, draws):
    return {"a": a, "b": b, "wins": wins, "losses": losses, "draws": draws}


class TestFit:
    def test_a_75_percent_score_is_about_191_elo(self):
        r = ratings.fit([match("x", "anchor", 75, 25, 0)], anchor="anchor")
        assert r["anchor"] == 0.0
        assert abs(r["x"] - 190.85) < 1.0

    def test_draws_count_as_half_a_point(self):
        r = ratings.fit([match("x", "anchor", 100, 0, 100)], anchor="anchor")  # 75% score
        assert abs(r["x"] - 190.85) < 1.0

    def test_ratings_combine_across_opponents(self):
        games = [match("b", "anchor", 75, 25, 0), match("c", "b", 75, 25, 0)]
        r = ratings.fit(games, anchor="anchor")
        assert abs(r["c"] - 2 * 190.85) < 2.0

    def test_a_perfect_score_stays_finite(self):
        r = ratings.fit([match("x", "anchor", 20, 0, 0)], anchor="anchor")
        assert 300 < r["x"] < 2000

    def test_the_anchor_must_have_played(self):
        with pytest.raises(ValueError):
            ratings.fit([match("x", "y", 5, 5, 0)], anchor="anchor")


class TestBootstrap:
    def test_standard_errors_shrink_with_more_games(self):
        small = ratings.bootstrap_se([match("x", "anchor", 15, 5, 0)], "anchor",
                                     n_boot=200, rng=np.random.default_rng(0))
        large = ratings.bootstrap_se([match("x", "anchor", 150, 50, 0)], "anchor",
                                     n_boot=200, rng=np.random.default_rng(0))
        assert small["anchor"] == 0.0
        assert large["x"] < 0.5 * small["x"]
        # Binomial: p = 0.75, n = 200 gives about 28 Elo.
        assert 20 < large["x"] < 40


class TestOpponents:
    def test_new_engine_plays_the_anchor_and_a_spread_of_ratings(self):
        r = {"anchor": 0, "a": -100, "b": 50, "c": 120, "d": 300, "e": 310}
        chosen = ratings.choose_opponents(r, "anchor", k=4)
        assert chosen[0] == "anchor"
        assert len(chosen) == 4 and len(set(chosen)) == 4
        assert "a" in chosen and ("d" in chosen or "e" in chosen)

    def test_small_pools_play_everyone(self):
        assert sorted(ratings.choose_opponents({"anchor": 0, "a": 10}, "anchor", k=4)) == ["a", "anchor"]
