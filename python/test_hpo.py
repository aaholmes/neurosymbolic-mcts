"""Tests for the hyperparameter search: 1-hour continuous training runs rated
jointly by Elo, with a Gaussian-process optimizer proposing the next run."""
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import hpo


@pytest.fixture
def tmp():
    d = tempfile.mkdtemp(prefix="hpo_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def in_space(params):
    for name, (lo, hi, _log, kind) in hpo.SPACE.items():
        v = params[name]
        assert lo <= v <= hi, (name, v)
        if kind is int:
            assert isinstance(v, int)
    return True


class TestStartingTrials:
    def test_defaults_come_first_and_are_repeated_with_another_seed(self):
        trials = hpo.starting_trials(n_sobol=5, seed=0)
        assert trials[0]["name"] == hpo.ANCHOR
        assert trials[0]["params"] == hpo.DEFAULTS
        repeats = [t for t in trials
                   if t["params"] == hpo.DEFAULTS and not t.get("epsilon")]
        assert len(repeats) == 2 and repeats[0]["seed"] != repeats[1]["seed"]

    def test_sobol_points_fill_the_space(self):
        trials = hpo.starting_trials(n_sobol=5, seed=0)
        sobol = [t for t in trials if t["name"].startswith("sobol")]
        assert len(sobol) == 5
        assert all(in_space(t["params"]) for t in trials)
        assert len({json.dumps(t["params"], sort_keys=True) for t in sobol}) == 5
        assert len({t["name"] for t in trials}) == len(trials)


class TestTrialConfig:
    def test_one_hour_network_only_run_with_exact_endings_and_a_decaying_rate(self):
        params = dict(hpo.DEFAULTS, lr=0.01, replay_ratio=2.0, simulations_per_move=150)
        cfg = hpo.trial_config(params, seed=3, work_dir="/r/t1", syzygy_path="/tb",
                               self_play_bin="/bin/sp")
        assert (cfg.lr, cfg.replay_ratio, cfg.simulations_per_move) == (0.01, 2.0, 150)
        assert cfg.max_seconds == 3600 and cfg.lr_schedule == "cosine"
        assert not cfg.enable_tier1 and not cfg.enable_material
        assert cfg.syzygy_path == "/tb" and cfg.mate_checks == 5
        assert (cfg.seed, cfg.work_dir, cfg.self_play_bin) == (3, "/r/t1", "/bin/sp")
        # 1,024 games in flight: 8% slower self-play than 1,920, but data about half as old.
        assert cfg.concurrent_games == 1024


class TestMatches:
    def test_command_plays_both_networks_in_the_training_mode(self):
        cmd = hpo.match_command("/w/a.pth", "/w/b.pth", games=40, sims=200, seed_offset=7,
                                syzygy_path="/tb", evaluate_bin="/bin/ev")
        assert cmd[:5] == ["/bin/ev", "/w/a.pth", "/w/b.pth", "40", "200"]
        for flag in ["--disable-tier1", "--disable-material", "--no-save-training-data"]:
            assert flag in cmd
        assert cmd[cmd.index("--syzygy") + 1] == "/tb"
        assert cmd[cmd.index("--seed-offset") + 1] == "7"

    def test_parse_result(self):
        out = "...\nWINS=12 LOSSES=20 DRAWS=8 WINRATE=0.4 ACCEPTED=false\n"
        assert hpo.parse_match(out) == (12, 20, 8)
        with pytest.raises(ValueError):
            hpo.parse_match("crashed")


class TestSuggest:
    def test_suggestion_lies_in_the_space(self):
        rng = np.random.default_rng(0)
        observed = [(t["params"], float(rng.normal())) for t in hpo.starting_trials(5, 0)]
        params = hpo.suggest(observed, seed=0)
        assert set(params) == set(hpo.SPACE)
        assert in_space(params)


class TestSearch:
    def test_trials_are_trained_rated_and_resumed(self, tmp):
        strength = {"lr": 3000.0}  # fake engines: stronger with a higher learning rate
        trained, played = [], []

        def train(trial, work_dir):
            trained.append(trial["name"])
            return os.path.join(work_dir, "cont_9.pth")

        def play(a, b, games, seed_offset):
            played.append((a["name"], b["name"], games))
            diff = strength["lr"] * (a["params"]["lr"] - b["params"]["lr"])
            p = 1 / (1 + 10 ** (-diff / 400))
            wins = int(round(p * games))
            return wins, games - wins, 0

        search = hpo.Search(tmp, train=train, play=play, games_per_engine=120, opponents=4)
        trials = hpo.starting_trials(n_sobol=3, seed=0)
        search.run_trials(trials[:3])
        assert trained == [t["name"] for t in trials[:3]]
        assert played[0][:2] == (trials[1]["name"], hpo.ANCHOR)

        again = hpo.Search(tmp, train=train, play=play, games_per_engine=120, opponents=4)
        again.run_trials(trials)  # the first three are already done
        assert trained[3:] == [t["name"] for t in trials[3:]]

        table = again.ratings()
        assert table[hpo.ANCHOR]["elo"] == 0.0
        best = max(trials, key=lambda t: t["params"]["lr"])
        assert max(table, key=lambda n: table[n]["elo"]) == best["name"]
        assert all(table[n]["se"] >= 0 for n in table)
        assert os.path.exists(os.path.join(tmp, "trials.jsonl"))
        assert os.path.exists(os.path.join(tmp, "matches.jsonl"))


def test_a_trial_whose_training_fails_is_not_recorded(tmp):
    def train(trial, work_dir):
        raise RuntimeError("self-play stalled")

    search = hpo.Search(tmp, train=train, play=lambda *a: (1, 1, 1))
    with pytest.raises(RuntimeError):
        search.run_trials([{"name": hpo.ANCHOR, "params": dict(hpo.DEFAULTS), "seed": 0}])
    assert search.trials() == {}


class TestSixSettings:
    def test_the_space_tunes_exploration_and_window_but_not_root_noise(self):
        assert set(hpo.SPACE) == {"lr", "replay_ratio", "simulations_per_move", "tau_decay",
                                  "c_puct", "window"}
        assert hpo.SPACE["c_puct"][:3] == (1.0, 3.0, True)
        assert hpo.SPACE["window"][:3] == (100_000, 1_000_000, True)
        assert hpo.SPACE["window"][3] is int
        assert hpo.SPACE["simulations_per_move"][:2] == (100, 400)
        assert hpo.DEFAULTS["c_puct"] == 1.414 and hpo.DEFAULTS["window"] == 200_000

    def test_six_sobol_points_and_the_root_noise_check_stay_in_the_starting_trials(self):
        trials = hpo.starting_trials()
        assert sum(t["name"].startswith("sobol") for t in trials) == 6
        noise = [t for t in trials if t["name"] == "t02_root_noise"]
        assert noise and noise[0]["epsilon"] == 0.25

    def test_trial_config_sets_exploration_window_and_root_noise(self):
        params = dict(hpo.DEFAULTS, c_puct=2.0, window=500_000)
        cfg = hpo.trial_config(params, 0, "/r", "/tb", "/bin/sp")
        assert (cfg.c_puct, cfg.window, cfg.dirichlet_epsilon) == (2.0, 500_000, 0.0)
        cfg = hpo.trial_config(params, 0, "/r", "/tb", "/bin/sp", dirichlet_epsilon=0.25)
        assert cfg.dirichlet_epsilon == 0.25

    def test_each_engine_plays_with_its_own_exploration_constant(self):
        cmd = hpo.match_command("/w/a.pth", "/w/b.pth", 40, 200, 7, "/tb", "/bin/ev",
                                a_c_puct=2.5, b_c_puct=1.2)
        assert cmd[cmd.index("--candidate-c-puct") + 1] == "2.5"
        assert cmd[cmd.index("--current-c-puct") + 1] == "1.2"
