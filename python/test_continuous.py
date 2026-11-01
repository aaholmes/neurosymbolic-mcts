"""Tests for continuous training: self-play runs nonstop with the newest
published weights, and the trainer trains on a sliding window of finished
games at a fixed replay ratio."""

import os
import sys
import shutil
import tempfile

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))

from model import SeresNet
from replay_buffer import SAMPLE_SIZE_FLOATS
import continuous as cont


def write_game(path, n, rng=None):
    rng = rng or np.random.default_rng(0)
    data = rng.standard_normal((n, SAMPLE_SIZE_FLOATS)).astype(np.float32)
    data.tofile(path)


@pytest.fixture
def tmp():
    d = tempfile.mkdtemp(prefix="cont_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


class TestStepsOwed:
    def test_nothing_generated_owes_nothing(self):
        assert cont.steps_owed(0, 0, replay_ratio=4.0, batch_size=64) == 0

    def test_owed_steps_follow_the_replay_ratio(self):
        # 1000 positions, each trained on 4 times, in batches of 100: 40 steps.
        assert cont.steps_owed(1000, 0, replay_ratio=4.0, batch_size=100) == 40
        assert cont.steps_owed(1000, 30, replay_ratio=4.0, batch_size=100) == 10

    def test_never_negative(self):
        assert cont.steps_owed(100, 50, replay_ratio=1.0, batch_size=64) == 0


class TestPause:
    def test_pauses_when_too_far_behind(self):
        assert cont.should_pause(False, owed=101, max_owed=100)

    def test_stays_paused_until_caught_up(self):
        assert cont.should_pause(True, owed=50, max_owed=100)
        assert not cont.should_pause(True, owed=0, max_owed=100)

    def test_runs_when_close_enough(self):
        assert not cont.should_pause(False, owed=100, max_owed=100)


class TestPublished:
    def test_pointer_format_matches_the_rust_reader(self, tmp):
        pointer = os.path.join(tmp, "published.txt")
        cont.write_published(pointer, 3, "/w/cont_3.pth")
        with open(pointer) as f:
            assert f.read() == "3 /w/cont_3.pth\n"
        assert os.listdir(tmp) == ["published.txt"]

    def test_game_version_from_filename(self):
        assert cont.game_version("game_17_4_v12.bin") == 12
        assert cont.game_version("/d/game_17_4.bin") is None


class TestGameWindow:
    def test_ingest_moves_finished_games_into_the_window(self, tmp):
        games = os.path.join(tmp, "games")
        os.makedirs(games)
        write_game(os.path.join(games, "game_1_0_v0.bin"), 10)
        write_game(os.path.join(games, "game_1_1_v2.bin"), 5)
        with open(os.path.join(games, "game_1_2_v2.bin.tmp"), "wb") as f:
            f.write(b"partial")
        window = cont.GameWindow(os.path.join(tmp, "window"), capacity=1000)
        added, versions = window.ingest(games)
        assert added == 15
        assert sorted(versions) == [0, 2]
        assert window.positions() == 15
        assert os.listdir(games) == ["game_1_2_v2.bin.tmp"]
        assert window.ingest(games) == (0, [])

    def test_window_drops_the_oldest_games(self, tmp):
        games = os.path.join(tmp, "games")
        os.makedirs(games)
        window = cont.GameWindow(os.path.join(tmp, "window"), capacity=25)
        for i in range(4):
            write_game(os.path.join(games, f"game_{i}_0_v0.bin"), 10, np.random.default_rng(i))
            window.ingest(games)
        assert window.positions() == 20

    def test_sample_returns_a_batch(self, tmp):
        games = os.path.join(tmp, "games")
        os.makedirs(games)
        write_game(os.path.join(games, "game_1_0_v0.bin"), 10)
        window = cont.GameWindow(os.path.join(tmp, "window"), capacity=1000)
        window.ingest(games)
        boards, scalars, values, policies = window.sample(8, np.random.default_rng(0))
        assert boards.shape == (8, 17, 8, 8)
        assert scalars.shape == (8, 2)
        assert values.shape == (8, 1)
        assert policies.shape == (8, 4672)


class TestTrainer:
    def _window(self, tmp, n=40):
        games = os.path.join(tmp, "games")
        os.makedirs(games, exist_ok=True)
        write_game(os.path.join(games, "game_1_0_v0.bin"), n)
        window = cont.GameWindow(os.path.join(tmp, "window"), capacity=1000)
        window.ingest(games)
        return window

    def _trainer(self, tmp, keep=2):
        return cont.Trainer(SeresNet(num_blocks=1, hidden_dim=16), os.path.join(tmp, "weights"),
                            optimizer="adam", lr=1e-3, batch_size=8, keep_published=keep,
                            device="cpu", seed=0)

    def test_train_steps_change_the_weights(self, tmp):
        window = self._window(tmp)
        trainer = self._trainer(tmp)
        before = [p.detach().clone() for p in trainer.model.parameters()]
        losses = trainer.train_steps(3, window)
        assert trainer.steps == 3
        assert set(losses) >= {"loss", "policy_loss", "value_loss"}
        assert any(not torch.equal(a, b) for a, b in zip(before, trainer.model.parameters()))

    def test_publish_writes_loadable_weights_and_moves_the_pointer(self, tmp):
        trainer = self._trainer(tmp)
        path0 = trainer.publish()
        path1 = trainer.publish()
        assert trainer.version == 1
        assert os.path.basename(path1) == "cont_1.pth"
        with open(trainer.pointer) as f:
            assert f.read() == f"1 {os.path.abspath(path1)}\n"
        ckpt = torch.load(path1, weights_only=False)
        model = SeresNet(num_blocks=1, hidden_dim=16)
        model.load_state_dict(ckpt["model_state_dict"])
        assert ckpt["global_minibatch"] == 0
        assert os.path.exists(path0)

    def test_publish_keeps_only_the_newest_weights(self, tmp):
        trainer = self._trainer(tmp, keep=2)
        paths = [trainer.publish() for _ in range(4)]
        assert [os.path.exists(p) for p in paths] == [False, False, True, True]

    def test_resume_restores_steps_and_version(self, tmp):
        window = self._window(tmp)
        trainer = self._trainer(tmp)
        trainer.train_steps(2, window)
        path = trainer.publish()
        again = self._trainer(tmp)
        again.resume(path)
        assert (again.steps, again.version) == (2, 0)
        assert again.publish().endswith("cont_1.pth")


class TestSelfPlayCommand:
    def test_command_watches_the_pointer_and_plays_many_games_in_parallel(self):
        cfg = cont.ContinuousConfig(concurrent_games=96, simulations_per_move=200,
                                    mate_checks=7, syzygy_path="/tb")
        cmd = cont.self_play_command(cfg, "/w/published.txt", "/w/cont_0.pth", "/d/games")
        assert cmd[cmd.index("--watch") + 1] == "/w/published.txt"
        assert cmd[cmd.index("--threads") + 1] == "96"
        assert "/d/games" in cmd and "/w/cont_0.pth" in cmd and "200" in cmd
        assert cmd[cmd.index("--mate-checks") + 1] == "7"
        assert cmd[cmd.index("--syzygy") + 1] == "/tb"


FAKE_SELF_PLAY = r'''
import os, sys, time
import numpy as np
games, n = sys.argv[1], int(sys.argv[2])
for i in range(n):
    data = np.random.default_rng(i).standard_normal((20, 5763)).astype(np.float32)
    tmp = os.path.join(games, f"game_0_{i}_v0.bin.tmp")
    data.tofile(tmp)
    os.rename(tmp, tmp[:-4])
time.sleep(600)
'''


class TestRun:
    def test_run_trains_publishes_and_stops_self_play(self, tmp):
        script = os.path.join(tmp, "fake_self_play.py")
        with open(script, "w") as f:
            f.write(FAKE_SELF_PLAY)
        cfg = cont.ContinuousConfig(
            work_dir=os.path.join(tmp, "run"), num_blocks=1, hidden_dim=16,
            optimizer="adam", lr=1e-3, batch_size=8, replay_ratio=2.0,
            min_positions=40, publish_every=5, max_steps=10, window=1000,
            device="cpu", poll_seconds=0.05,
        )
        games_dir = os.path.join(cfg.work_dir, "games")
        procs = []

        def launch(cmd_cfg, pointer, first, games):
            assert games == games_dir
            assert os.path.exists(pointer)
            p = __import__("subprocess").Popen([sys.executable, script, games, "5"])
            procs.append(p)
            return p

        summary = cont.run(cfg, launch=launch)
        assert summary["steps"] == 10
        # 100 positions x 2 / 8 = 25 steps owed; it stops at 10.
        assert summary["positions_generated"] == 100
        assert summary["version"] == 2  # v0 at start, then every 5 steps
        assert procs[0].poll() is not None, "self-play must be stopped"
        log = os.path.join(cfg.work_dir, "continuous_log.jsonl")
        lines = open(log).read().splitlines()
        assert len(lines) == 2
