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

    def test_a_replay_ratio_below_one_trains_on_a_fraction_of_positions(self):
        assert cont.steps_owed(1000, 0, replay_ratio=0.25, batch_size=100) == 2
        assert cont.steps_owed(1000, 2, replay_ratio=0.25, batch_size=100) == 0


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
        assert "--syzygy-set" not in cmd

    def test_the_table_set_reaches_self_play(self):
        cfg = cont.ContinuousConfig(syzygy_path="/tb", syzygy_set="6+")
        cmd = cont.self_play_command(cfg, "/w/published.txt", "/w/cont_0.pth", "/d/games")
        assert cmd[cmd.index("--syzygy-set") + 1] == "6+"


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


def write_valid_game(path, n, rng=None):
    """Random positions with policy targets that are probability distributions
    and values in [-1, 1], so the loss is finite."""
    rng = rng or np.random.default_rng(0)
    data = rng.standard_normal((n, SAMPLE_SIZE_FLOATS)).astype(np.float32)
    policy = rng.random((n, 4672)).astype(np.float32) ** 8
    data[:, -4672:] = policy / policy.sum(axis=1, keepdims=True)
    data[:, -4673] = rng.choice([-1.0, 0.0, 1.0], size=n)
    data.tofile(path)


class TestCheaperSteps:
    """bf16 autocast and channels-last layout make the trainer's GPU step
    cheaper; the weights stay f32 and are published in the layout Rust reads."""

    def _window(self, tmp, n=40):
        games = os.path.join(tmp, "games")
        os.makedirs(games, exist_ok=True)
        write_valid_game(os.path.join(games, "game_1_0_v0.bin"), n)
        window = cont.GameWindow(os.path.join(tmp, "window"), capacity=1000)
        window.ingest(games)
        return window

    def _trainer(self, tmp, **kw):
        torch.manual_seed(0)
        return cont.Trainer(SeresNet(num_blocks=1, hidden_dim=16), os.path.join(tmp, "weights"),
                            optimizer="muon", lr=0.02, batch_size=8, device="cpu", seed=0, **kw)

    def test_unknown_precision_is_rejected(self, tmp):
        with pytest.raises(ValueError):
            self._trainer(tmp, precision="f8")

    def test_bf16_steps_train_and_keep_f32_weights(self, tmp):
        window = self._window(tmp)
        trainer = self._trainer(tmp, precision="bf16")
        before = [p.detach().clone() for p in trainer.model.parameters()]
        losses = trainer.train_steps(3, window)
        assert np.isfinite(losses["loss"])
        assert all(p.dtype == torch.float32 for p in trainer.model.parameters())
        assert any(not torch.equal(a, b) for a, b in zip(before, trainer.model.parameters()))

    def test_bf16_loss_is_close_to_f32_loss(self, tmp):
        window = self._window(tmp)
        f32 = self._trainer(tmp, precision="f32").train_steps(1, window)["loss"]
        bf16 = self._trainer(tmp, precision="bf16").train_steps(1, window)["loss"]
        assert abs(bf16 - f32) < 0.02 * abs(f32)

    def test_channels_last_matches_the_default_layout(self, tmp):
        window = self._window(tmp)
        a = self._trainer(tmp).train_steps(2, window)["loss"]
        b = self._trainer(tmp, channels_last=True).train_steps(2, window)["loss"]
        assert abs(a - b) < 1e-4 * abs(a)

    def test_channels_last_publishes_contiguous_weights(self, tmp):
        window = self._window(tmp)
        trainer = self._trainer(tmp, channels_last=True)
        trainer.train_steps(1, window)
        ckpt = torch.load(trainer.publish(), weights_only=False)
        assert all(t.is_contiguous() for t in ckpt["model_state_dict"].values())

    def test_config_reaches_the_trainer(self, tmp):
        cfg = cont.ContinuousConfig(precision="bf16", channels_last=True)
        assert (cfg.precision, cfg.channels_last) == ("bf16", True)
        args = cont.parse_args(["--precision", "bf16", "--channels-last", "true"])
        assert (args.precision, args.channels_last) == ("bf16", True)
        trainer = cont.make_trainer(cont.ContinuousConfig(
            num_blocks=1, hidden_dim=16, device="cpu", precision="bf16", channels_last=True),
            os.path.join(tmp, "w"))
        assert (trainer.precision, trainer.channels_last) == ("bf16", True)


def test_continuous_training_defaults_to_bf16_channels_last():
    cfg = cont.ContinuousConfig()
    assert (cfg.precision, cfg.channels_last) == ("bf16", True)


def test_self_play_command_passes_the_exploration_settings():
    cfg = cont.ContinuousConfig(tau0=0.9, tau_decay=0.8, tau_min=0.05,
                                dirichlet_alpha=0.3, dirichlet_epsilon=0.25)
    cmd = cont.self_play_command(cfg, "/w/p.txt", "/w/cont_0.pth", "/d/games")
    for flag, value in [("--tau0", "0.9"), ("--tau-decay", "0.8"), ("--tau-min", "0.05"),
                        ("--dirichlet-alpha", "0.3"), ("--dirichlet-epsilon", "0.25")]:
        assert cmd[cmd.index(flag) + 1] == value
    assert "--explore-base" not in cmd


class TestLearningRateSchedule:
    def test_cosine_falls_from_one_to_zero(self):
        assert cont.lr_factor("cosine", 0.0) == 1.0
        assert abs(cont.lr_factor("cosine", 0.5) - 0.5) < 1e-12
        assert abs(cont.lr_factor("cosine", 1.0)) < 1e-12
        assert abs(cont.lr_factor("cosine", 2.0)) < 1e-12

    def test_constant_stays_at_one(self):
        assert cont.lr_factor("constant", 0.7) == 1.0

    def test_unknown_schedule_is_rejected(self):
        with pytest.raises(ValueError):
            cont.lr_factor("linear", 0.5)

    def test_progress_uses_the_time_limit_or_else_the_step_limit(self):
        cfg = cont.ContinuousConfig(max_seconds=100)
        assert cont.run_progress(cfg, elapsed=25, steps=0) == 0.25
        cfg = cont.ContinuousConfig(max_steps=200)
        assert cont.run_progress(cfg, elapsed=25, steps=50) == 0.25
        assert cont.run_progress(cont.ContinuousConfig(), elapsed=25, steps=50) == 0.0

    def test_trainer_scales_every_learning_rate_including_muons_backend(self, tmp):
        trainer = cont.Trainer(SeresNet(num_blocks=1, hidden_dim=16), os.path.join(tmp, "w"),
                               optimizer="muon", lr=0.02, batch_size=8, device="cpu")
        trainer.set_lr_factor(0.5)
        assert trainer.optimizer.param_groups[0]["lr"] == 0.01
        assert abs(trainer.optimizer.backend.param_groups[0]["lr"] - 0.001) < 1e-12
        trainer.set_lr_factor(1.0)
        assert trainer.optimizer.param_groups[0]["lr"] == 0.02


class TestRunLimits:
    def test_self_play_binary_replaces_cargo_run(self):
        cfg = cont.ContinuousConfig(self_play_bin="/bin/sp")
        cmd = cont.self_play_command(cfg, "/w/p.txt", "/w/cont_0.pth", "/d/games")
        assert cmd[0] == "/bin/sp" and "cargo" not in cmd

    def test_run_stops_at_the_time_limit_and_publishes_the_final_weights(self, tmp):
        script = os.path.join(tmp, "fake_self_play.py")
        with open(script, "w") as f:
            f.write(FAKE_SELF_PLAY)
        cfg = cont.ContinuousConfig(
            work_dir=os.path.join(tmp, "run"), num_blocks=1, hidden_dim=16,
            optimizer="adam", lr=1e-3, batch_size=8, replay_ratio=2.0,
            min_positions=40, publish_every=1000, max_seconds=3, window=1000,
            device="cpu", poll_seconds=0.05, lr_schedule="cosine",
        )

        def launch(cmd_cfg, pointer, first, games):
            return __import__("subprocess").Popen([sys.executable, script, games, "5"])

        summary = cont.run(cfg, launch=launch)
        assert summary["steps"] == 25  # 100 positions x 2 / 8, then idle until the limit
        assert summary["version"] == 1  # v0 at start, the final weights at the end
        assert os.path.basename(summary["weights"]) == "cont_1.pth"
        assert torch.load(summary["weights"], weights_only=False)["global_minibatch"] == 25


class TestStall:
    def test_run_raises_when_no_game_finishes_for_stall_seconds(self, tmp):
        script = os.path.join(tmp, "fake_self_play.py")
        with open(script, "w") as f:
            f.write(FAKE_SELF_PLAY)  # 5 games, then nothing
        cfg = cont.ContinuousConfig(
            work_dir=os.path.join(tmp, "run"), num_blocks=1, hidden_dim=16,
            optimizer="adam", lr=1e-3, batch_size=8, replay_ratio=2.0,
            min_positions=40, publish_every=1000, max_seconds=60, window=1000,
            device="cpu", poll_seconds=0.05, stall_seconds=2,
        )
        procs = []

        def launch(cmd_cfg, pointer, first, games):
            p = __import__("subprocess").Popen([sys.executable, script, games, "5"])
            procs.append(p)
            return p

        start = __import__("time").time()
        with pytest.raises(cont.SelfPlayStalled):
            cont.run(cfg, launch=launch)
        assert __import__("time").time() - start < 20, "must not wait for the time limit"
        assert procs[0].poll() is not None, "self-play must be stopped"

    def test_stall_check_defaults_to_five_minutes(self):
        assert cont.ContinuousConfig().stall_seconds == 300

    def test_stall_check_ignores_time_spent_paused(self):
        # Paused for 400 s of the last 500 s without a game: 100 s running, no stall.
        assert not cont.stalled(now=1500.0, last_game=1000.0, paused_s=400.0, stall_seconds=300)
        assert cont.stalled(now=1500.0, last_game=1000.0, paused_s=100.0, stall_seconds=300)
        assert not cont.stalled(now=1500.0, last_game=1000.0, paused_s=0.0, stall_seconds=0)


def test_self_play_command_passes_the_exploration_constant():
    assert cont.ContinuousConfig().c_puct == 1.414
    cmd = cont.self_play_command(cont.ContinuousConfig(c_puct=2.2), "/w/p.txt", "/w/c.pth", "/d/g")
    assert cmd[cmd.index("--c-puct") + 1] == "2.2"


class TestSidePositions:
    """Side positions have a policy target and no value target (NaN)."""

    def _model(self):
        return cont.ContinuousConfig(num_blocks=1, hidden_dim=16, device="cpu", optimizer="adam",
                                     precision="f32", channels_last=False)

    def test_value_loss_skips_positions_without_a_value_target(self, tmp):
        from train import batch_loss
        cfg = self._model()
        trainer = cont.make_trainer(cfg, os.path.join(tmp, "w"))
        rng = np.random.default_rng(0)
        n = 8
        boards = torch.from_numpy(rng.standard_normal((n, 17, 8, 8)).astype(np.float32))
        scalars = torch.zeros((n, 2))
        policies = torch.full((n, 4672), 1.0 / 4672)
        values = torch.from_numpy(rng.uniform(-1, 1, (n, 1)).astype(np.float32))
        masked = values.clone()
        masked[4:] = float("nan")
        _, p_all, v_all, _ = batch_loss(trainer.model, boards, scalars, values, policies, "cpu")
        loss, p_m, v_m, _ = batch_loss(trainer.model, boards, scalars, masked, policies, "cpu")
        _, _, v_first, _ = batch_loss(trainer.model, boards[:4], scalars[:4], values[:4],
                                      policies[:4], "cpu")
        assert torch.isfinite(loss)
        assert torch.allclose(p_m, p_all)  # every position keeps its policy target
        assert torch.allclose(v_m, v_first)  # value loss averaged over the 4 with a target
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in trainer.model.parameters()
                   if p.grad is not None)

    def test_a_batch_of_only_side_positions_trains_the_policy_alone(self, tmp):
        from train import batch_loss
        trainer = cont.make_trainer(self._model(), os.path.join(tmp, "w"))
        n = 4
        boards = torch.randn((n, 17, 8, 8))
        values = torch.full((n, 1), float("nan"))
        loss, p, v, _ = batch_loss(trainer.model, boards, torch.zeros((n, 2)), values,
                                   torch.full((n, 4672), 1.0 / 4672), "cpu")
        assert torch.isfinite(loss) and v.item() == 0.0
        assert torch.allclose(loss, p)

    def test_symmetry_transforms_keep_a_missing_value_missing(self):
        from augmentation import random_transform
        rng = np.random.default_rng(0)
        board = np.zeros((17, 8, 8), dtype=np.float32)  # no pawns, no castling: 8 images
        board[5, 0, 4] = 1.0
        board[11, 7, 3] = 1.0
        policy = np.full(4672, 1.0 / 4672, dtype=np.float32)
        for _ in range(16):
            _, _, value, _ = random_transform(board, 0.0, float("nan"), policy, rng)
            assert np.isnan(value)

    def test_self_play_command_passes_the_side_position_settings(self):
        assert cont.ContinuousConfig().side_fraction == 0.0
        assert cont.ContinuousConfig().side_two_moves == 0.5
        cmd = cont.self_play_command(cont.ContinuousConfig(side_fraction=0.5, side_two_moves=0.25),
                                     "/w/p.txt", "/w/c.pth", "/d/g")
        assert cmd[cmd.index("--side-fraction") + 1] == "0.5"
        assert cmd[cmd.index("--side-two-moves") + 1] == "0.25"


class TestPolicyOnly:
    """The policy-only engine: leaf values from the classical search alone, so
    the network trains only its policy."""

    def test_self_play_runs_the_policy_only_search(self):
        assert not cont.ContinuousConfig().policy_only
        cmd = cont.self_play_command(cont.ContinuousConfig(policy_only=True),
                                     "/w/p.txt", "/w/c.pth", "/d/g")
        assert cmd[cmd.index("--policy-only") + 1] == "revisit"
        plain = cont.self_play_command(cont.ContinuousConfig(), "/w/p.txt", "/w/c.pth", "/d/g")
        assert "--policy-only" not in plain

    def test_a_value_weight_of_zero_trains_the_policy_alone(self, tmp):
        from train import batch_loss
        cfg = cont.ContinuousConfig(num_blocks=1, hidden_dim=16, device="cpu", optimizer="adam",
                                    precision="f32", channels_last=False)
        trainer = cont.make_trainer(cfg, os.path.join(tmp, "w"))
        n = 4
        boards = torch.randn((n, 17, 8, 8))
        values = torch.ones((n, 1))
        loss, p, v, _ = batch_loss(trainer.model, boards, torch.zeros((n, 2)), values,
                                   torch.full((n, 4672), 1.0 / 4672), "cpu", value_weight=0.0)
        assert v.item() > 0.0  # still reported
        assert torch.allclose(loss, p)

    def test_the_policy_only_trainer_gives_the_value_no_weight(self, tmp):
        base = dict(num_blocks=1, hidden_dim=16, device="cpu", optimizer="adam",
                    precision="f32", channels_last=False)
        assert cont.make_trainer(cont.ContinuousConfig(**base), os.path.join(tmp, "a")).value_weight == 1.0
        policy = cont.make_trainer(cont.ContinuousConfig(policy_only=True, **base),
                                   os.path.join(tmp, "b"))
        assert policy.value_weight == 0.0


def test_the_quiescence_settings_reach_self_play():
    assert cont.ContinuousConfig().qsearch_config == ""
    plain = cont.self_play_command(cont.ContinuousConfig(), "/w/p.txt", "/w/c.pth", "/d/g")
    assert "--qsearch-config" not in plain
    cmd = cont.self_play_command(cont.ContinuousConfig(qsearch_config="default+tt"),
                                 "/w/p.txt", "/w/c.pth", "/d/g")
    assert cmd[cmd.index("--qsearch-config") + 1] == "default+tt"
