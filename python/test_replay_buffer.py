"""Tests for the FIFO Replay Buffer."""

import os
import json
import shutil
import tempfile
import struct
import numpy as np
import pytest

from replay_buffer import ReplayBuffer

# Match train.py constants
INPUT_CHANNELS = 17
BOARD_SIZE = 64
POLICY_SIZE = 4672
SAMPLE_SIZE_FLOATS = (INPUT_CHANNELS * BOARD_SIZE) + 1 + 1 + POLICY_SIZE  # 5762


def make_fake_bin(path, num_positions):
    """Create a fake .bin file with `num_positions` samples of the correct size."""
    data = np.random.randn(num_positions, SAMPLE_SIZE_FLOATS).astype(np.float32)
    data.tofile(path)


@pytest.fixture
def tmp_dirs():
    """Create temporary directories for buffer and source games."""
    buffer_dir = tempfile.mkdtemp(prefix="buffer_")
    source_dir = tempfile.mkdtemp(prefix="source_")
    yield buffer_dir, source_dir
    shutil.rmtree(buffer_dir, ignore_errors=True)
    shutil.rmtree(source_dir, ignore_errors=True)


class TestReplayBuffer:
    def test_empty_buffer_has_zero_positions(self, tmp_dirs):
        buffer_dir, _ = tmp_dirs
        buf = ReplayBuffer(capacity_positions=10000, buffer_dir=buffer_dir)
        assert buf.total_positions() == 0

    def test_add_games_updates_position_count(self, tmp_dirs):
        buffer_dir, source_dir = tmp_dirs
        make_fake_bin(os.path.join(source_dir, "game_1.bin"), 50)
        make_fake_bin(os.path.join(source_dir, "game_2.bin"), 30)

        buf = ReplayBuffer(capacity_positions=100000, buffer_dir=buffer_dir)
        added = buf.add_games(source_dir)
        assert added == 80
        assert buf.total_positions() == 80

    def test_add_games_copies_bin_files_to_buffer_dir(self, tmp_dirs):
        buffer_dir, source_dir = tmp_dirs
        make_fake_bin(os.path.join(source_dir, "game_1.bin"), 10)

        buf = ReplayBuffer(capacity_positions=100000, buffer_dir=buffer_dir)
        buf.add_games(source_dir)

        # Check files were copied into buffer_dir
        bin_files = [f for f in os.listdir(buffer_dir) if f.endswith(".bin")]
        assert len(bin_files) == 1

    def test_evict_oldest_removes_oldest_files(self, tmp_dirs):
        buffer_dir, source_dir = tmp_dirs

        # Add two batches with known ordering
        src1 = tempfile.mkdtemp(prefix="src1_")
        src2 = tempfile.mkdtemp(prefix="src2_")
        try:
            make_fake_bin(os.path.join(src1, "game_old.bin"), 50)
            make_fake_bin(os.path.join(src2, "game_new.bin"), 50)

            buf = ReplayBuffer(capacity_positions=60, buffer_dir=buffer_dir)
            buf.add_games(src1)
            buf.add_games(src2)

            # Over capacity (100 > 60), evict should remove oldest
            buf.evict_oldest()
            assert buf.total_positions() <= 60
            # Should have removed the old game
            assert buf.total_positions() == 50
        finally:
            shutil.rmtree(src1, ignore_errors=True)
            shutil.rmtree(src2, ignore_errors=True)

    def test_evict_respects_capacity(self, tmp_dirs):
        buffer_dir, source_dir = tmp_dirs

        # Create 5 games of 20 positions each = 100 total
        for i in range(5):
            src = tempfile.mkdtemp(prefix=f"src{i}_")
            make_fake_bin(os.path.join(src, f"game_{i}.bin"), 20)
            buf_if_first = ReplayBuffer(capacity_positions=50, buffer_dir=buffer_dir)
            if i == 0:
                buf = buf_if_first
            buf.add_games(src)
            shutil.rmtree(src, ignore_errors=True)

        # Load from scratch to test capacity
        buf2 = ReplayBuffer(capacity_positions=50, buffer_dir=buffer_dir)
        buf2.load_manifest()
        buf2.evict_oldest()
        assert buf2.total_positions() <= 50

    def test_sample_batch_returns_correct_shapes(self, tmp_dirs):
        buffer_dir, source_dir = tmp_dirs
        make_fake_bin(os.path.join(source_dir, "game_1.bin"), 100)

        buf = ReplayBuffer(capacity_positions=100000, buffer_dir=buffer_dir)
        buf.add_games(source_dir)

        batch_size = 8
        boards, materials, values, policies = buf.sample_batch(batch_size)
        assert boards.shape == (batch_size, INPUT_CHANNELS, 8, 8)
        assert materials.shape == (batch_size, 1)
        assert values.shape == (batch_size, 1)
        assert policies.shape == (batch_size, POLICY_SIZE)

    def test_sample_batch_raises_on_empty_buffer(self, tmp_dirs):
        buffer_dir, _ = tmp_dirs
        buf = ReplayBuffer(capacity_positions=10000, buffer_dir=buffer_dir)
        with pytest.raises(ValueError):
            buf.sample_batch(8)

    def test_manifest_save_and_load_roundtrip(self, tmp_dirs):
        buffer_dir, source_dir = tmp_dirs
        make_fake_bin(os.path.join(source_dir, "game_1.bin"), 25)
        make_fake_bin(os.path.join(source_dir, "game_2.bin"), 35)

        buf1 = ReplayBuffer(capacity_positions=100000, buffer_dir=buffer_dir)
        buf1.add_games(source_dir)
        buf1.save_manifest()

        # Load from manifest
        buf2 = ReplayBuffer(capacity_positions=100000, buffer_dir=buffer_dir)
        buf2.load_manifest()
        assert buf2.total_positions() == 60
        assert len(buf2.entries) == 2

    def test_fifo_ordering_preserved(self, tmp_dirs):
        buffer_dir, _ = tmp_dirs
        buf = ReplayBuffer(capacity_positions=100000, buffer_dir=buffer_dir)

        for i in range(3):
            src = tempfile.mkdtemp(prefix=f"src{i}_")
            make_fake_bin(os.path.join(src, f"game_{i}.bin"), 10)
            buf.add_games(src)
            shutil.rmtree(src, ignore_errors=True)

        # Entries should be in insertion order
        timestamps = [e["timestamp"] for e in buf.entries]
        assert timestamps == sorted(timestamps)


    def test_add_games_with_empty_source_dir(self, tmp_dirs):
        """add_games on a dir with no .bin files returns 0."""
        buffer_dir, source_dir = tmp_dirs
        buf = ReplayBuffer(capacity_positions=10000, buffer_dir=buffer_dir)
        added = buf.add_games(source_dir)
        assert added == 0
        assert buf.total_positions() == 0

    def test_add_games_skips_empty_bin_files(self, tmp_dirs):
        """add_games skips .bin files with 0 bytes."""
        buffer_dir, source_dir = tmp_dirs
        # Create an empty .bin file
        with open(os.path.join(source_dir, "empty.bin"), "w") as f:
            pass
        # Create a valid .bin file
        make_fake_bin(os.path.join(source_dir, "valid.bin"), 10)

        buf = ReplayBuffer(capacity_positions=100000, buffer_dir=buffer_dir)
        added = buf.add_games(source_dir)
        assert added == 10
        assert len(buf.entries) == 1

    def test_add_games_skips_undersized_bin_files(self, tmp_dirs):
        """add_games skips .bin files smaller than one sample."""
        buffer_dir, source_dir = tmp_dirs
        # Write a file with less data than BYTES_PER_SAMPLE
        with open(os.path.join(source_dir, "tiny.bin"), "wb") as f:
            f.write(b"\x00" * 100)

        buf = ReplayBuffer(capacity_positions=100000, buffer_dir=buffer_dir)
        added = buf.add_games(source_dir)
        assert added == 0
        assert len(buf.entries) == 0

    def test_sample_batch_larger_than_total(self, tmp_dirs):
        """sample_batch with batch_size > total_positions uses replacement."""
        buffer_dir, source_dir = tmp_dirs
        make_fake_bin(os.path.join(source_dir, "game.bin"), 3)

        buf = ReplayBuffer(capacity_positions=100000, buffer_dir=buffer_dir)
        buf.add_games(source_dir)

        boards, materials, values, policies = buf.sample_batch(10)
        assert boards.shape[0] == 10

    def test_sample_batch_single_position(self, tmp_dirs):
        """sample_batch works with buffer containing just 1 position."""
        buffer_dir, source_dir = tmp_dirs
        make_fake_bin(os.path.join(source_dir, "game.bin"), 1)

        buf = ReplayBuffer(capacity_positions=100000, buffer_dir=buffer_dir)
        buf.add_games(source_dir)

        boards, materials, values, policies = buf.sample_batch(5)
        assert boards.shape[0] == 5

    def test_evict_on_empty_buffer(self, tmp_dirs):
        """evict_oldest on empty buffer is a no-op."""
        buffer_dir, _ = tmp_dirs
        buf = ReplayBuffer(capacity_positions=10, buffer_dir=buffer_dir)
        buf.evict_oldest()  # Should not raise
        assert buf.total_positions() == 0

    def test_evict_handles_already_deleted_file(self, tmp_dirs):
        """evict_oldest handles FileNotFoundError when file is already gone."""
        buffer_dir, source_dir = tmp_dirs
        make_fake_bin(os.path.join(source_dir, "game.bin"), 50)

        buf = ReplayBuffer(capacity_positions=10, buffer_dir=buffer_dir)
        buf.add_games(source_dir)

        # Manually delete the file before evicting
        os.remove(buf.entries[0]["path"])
        buf.evict_oldest()  # Should not raise
        assert buf.total_positions() == 0

    def test_load_manifest_filters_missing_files(self, tmp_dirs):
        """load_manifest removes entries pointing to deleted files."""
        buffer_dir, source_dir = tmp_dirs
        make_fake_bin(os.path.join(source_dir, "game1.bin"), 20)
        make_fake_bin(os.path.join(source_dir, "game2.bin"), 30)

        buf = ReplayBuffer(capacity_positions=100000, buffer_dir=buffer_dir)
        buf.add_games(source_dir)
        buf.save_manifest()

        # Delete one file
        os.remove(buf.entries[0]["path"])

        buf2 = ReplayBuffer(capacity_positions=100000, buffer_dir=buffer_dir)
        buf2.load_manifest()
        assert len(buf2.entries) == 1
        assert buf2.total_positions() == 30

    def test_load_manifest_no_manifest_file(self, tmp_dirs):
        """load_manifest with no manifest.json keeps empty entries."""
        buffer_dir, _ = tmp_dirs
        buf = ReplayBuffer(capacity_positions=10000, buffer_dir=buffer_dir)
        buf.load_manifest()
        assert len(buf.entries) == 0

    def test_add_games_ignores_non_bin_files(self, tmp_dirs):
        """add_games only picks up .bin files, not other extensions."""
        buffer_dir, source_dir = tmp_dirs
        make_fake_bin(os.path.join(source_dir, "game.bin"), 10)
        with open(os.path.join(source_dir, "readme.txt"), "w") as f:
            f.write("not a bin file")
        with open(os.path.join(source_dir, "data.csv"), "w") as f:
            f.write("1,2,3")

        buf = ReplayBuffer(capacity_positions=100000, buffer_dir=buffer_dir)
        added = buf.add_games(source_dir)
        assert added == 10
        assert len(buf.entries) == 1

    def test_multiple_add_games_accumulates(self, tmp_dirs):
        """Multiple add_games calls accumulate positions."""
        buffer_dir, _ = tmp_dirs
        buf = ReplayBuffer(capacity_positions=100000, buffer_dir=buffer_dir)

        for i in range(3):
            src = tempfile.mkdtemp(prefix=f"batch{i}_")
            make_fake_bin(os.path.join(src, f"game.bin"), 10 * (i + 1))
            buf.add_games(src)
            shutil.rmtree(src, ignore_errors=True)

        assert buf.total_positions() == 10 + 20 + 30
        assert len(buf.entries) == 3

    def test_sample_batch_returns_finite_values(self, tmp_dirs):
        """Sampled data should contain finite floating point values."""
        buffer_dir, source_dir = tmp_dirs
        # Use controlled data instead of random
        data = np.ones((5, SAMPLE_SIZE_FLOATS), dtype=np.float32) * 0.5
        data.tofile(os.path.join(source_dir, "game.bin"))

        buf = ReplayBuffer(capacity_positions=100000, buffer_dir=buffer_dir)
        buf.add_games(source_dir)

        boards, materials, values, policies = buf.sample_batch(3)
        assert np.all(np.isfinite(boards))
        assert np.all(np.isfinite(materials))
        assert np.all(np.isfinite(values))
        assert np.all(np.isfinite(policies))


class TestEloWeighting:
    def test_same_elo_weighted_uniformly(self, tmp_dirs):
        """Entries with the same Elo should have equal weight per position."""
        buffer_dir, _ = tmp_dirs
        buf = ReplayBuffer(capacity_positions=100000, buffer_dir=buffer_dir)

        srcs = [tempfile.mkdtemp(prefix=f"src{i}_") for i in range(3)]
        try:
            for src in srcs:
                make_fake_bin(os.path.join(src, "game.bin"), 100)
                buf.add_games(src, model_elo=0.0)

            # All same Elo, same size → equal weights
            counts = np.array([e["num_positions"] for e in buf.entries], dtype=np.float64)
            elos = np.array([e.get("model_elo", 0.0) for e in buf.entries])
            max_elo = elos.max()
            expected_scores = 1.0 / (1.0 + np.power(10.0, (max_elo - elos) / 400.0))
            weights = counts * 2.0 * expected_scores
            weights /= weights.sum()

            assert weights[0] == pytest.approx(weights[1], rel=1e-6)
            assert weights[1] == pytest.approx(weights[2], rel=1e-6)
            assert weights[0] == pytest.approx(1.0 / 3.0, rel=1e-6)
        finally:
            for src in srcs:
                shutil.rmtree(src, ignore_errors=True)

    def test_higher_elo_weighted_more(self, tmp_dirs):
        """Data from a higher-Elo model should be weighted more than lower."""
        buffer_dir, _ = tmp_dirs
        buf = ReplayBuffer(capacity_positions=100000, buffer_dir=buffer_dir)

        src1 = tempfile.mkdtemp(prefix="low_")
        src2 = tempfile.mkdtemp(prefix="high_")
        try:
            make_fake_bin(os.path.join(src1, "low.bin"), 100)
            buf.add_games(src1, model_elo=0.0)
            make_fake_bin(os.path.join(src2, "high.bin"), 100)
            # 55% winrate → elo_delta = -400*log10(1/0.55 - 1) ≈ 35
            buf.add_games(src2, model_elo=35.0)

            counts = np.array([e["num_positions"] for e in buf.entries], dtype=np.float64)
            elos = np.array([e.get("model_elo", 0.0) for e in buf.entries])
            max_elo = elos.max()
            expected_scores = 1.0 / (1.0 + np.power(10.0, (max_elo - elos) / 400.0))
            weights = counts * 2.0 * expected_scores
            weights /= weights.sum()

            # 35 Elo gap → expected score ≈ 0.45 → weight ratio ≈ 0.9:1
            # So higher Elo should have more weight, but not dramatically more
            assert weights[1] > weights[0], "Higher Elo should have more weight"
            # The ratio should be moderate (~1.2:1), not extreme like 7:1
            ratio = weights[1] / weights[0]
            assert 1.0 < ratio < 2.0, f"Ratio {ratio:.2f} should be moderate (1-2x)"
        finally:
            shutil.rmtree(src1, ignore_errors=True)
            shutil.rmtree(src2, ignore_errors=True)

    def test_large_elo_gap_still_contributes(self, tmp_dirs):
        """Even with a 500 Elo gap, old data should still have non-zero weight."""
        buffer_dir, _ = tmp_dirs
        buf = ReplayBuffer(capacity_positions=100000, buffer_dir=buffer_dir)

        src1 = tempfile.mkdtemp(prefix="old_")
        src2 = tempfile.mkdtemp(prefix="new_")
        try:
            make_fake_bin(os.path.join(src1, "old.bin"), 100)
            buf.add_games(src1, model_elo=0.0)
            make_fake_bin(os.path.join(src2, "new.bin"), 100)
            buf.add_games(src2, model_elo=500.0)

            counts = np.array([e["num_positions"] for e in buf.entries], dtype=np.float64)
            elos = np.array([e.get("model_elo", 0.0) for e in buf.entries])
            max_elo = elos.max()
            expected_scores = 1.0 / (1.0 + np.power(10.0, (max_elo - elos) / 400.0))
            weights = counts * 2.0 * expected_scores
            weights /= weights.sum()

            # Old data should still get meaningful weight (~0.1)
            assert weights[0] > 0.05, f"Old data weight {weights[0]:.3f} should be > 0.05"
            # But new data should dominate
            assert weights[1] > weights[0]
        finally:
            shutil.rmtree(src1, ignore_errors=True)
            shutil.rmtree(src2, ignore_errors=True)

    def test_legacy_entries_default_to_elo_zero(self, tmp_dirs):
        """Entries without model_elo field default to 0.0."""
        buffer_dir, _ = tmp_dirs
        buf = ReplayBuffer(capacity_positions=100000, buffer_dir=buffer_dir)

        src = tempfile.mkdtemp(prefix="legacy_")
        try:
            make_fake_bin(os.path.join(src, "game.bin"), 50)
            buf.add_games(src)

            # Manually strip model_elo to simulate legacy manifest
            for entry in buf.entries:
                entry.pop("model_elo", None)
            buf.save_manifest()

            # Reload and sample — should not crash
            buf2 = ReplayBuffer(capacity_positions=100000, buffer_dir=buffer_dir)
            buf2.load_manifest()
            boards, _, _, _ = buf2.sample_batch(4)
            assert boards.shape[0] == 4
        finally:
            shutil.rmtree(src, ignore_errors=True)

    def test_elo_weighting_sample_batch_integration(self, tmp_dirs):
        """sample_batch with Elo weighting returns valid data."""
        buffer_dir, _ = tmp_dirs
        buf = ReplayBuffer(capacity_positions=100000, buffer_dir=buffer_dir)

        src = tempfile.mkdtemp(prefix="data_")
        try:
            make_fake_bin(os.path.join(src, "game.bin"), 20)
            buf.add_games(src, model_elo=100.0)

            boards, materials, values, policies = buf.sample_batch(8)
            assert boards.shape == (8, INPUT_CHANNELS, 8, 8)
            assert np.all(np.isfinite(boards))
        finally:
            shutil.rmtree(src, ignore_errors=True)

    def test_200_elo_gap_weight_ratio(self, tmp_dirs):
        """200 Elo gap should give ~0.48x weight ratio."""
        buffer_dir, _ = tmp_dirs
        buf = ReplayBuffer(capacity_positions=100000, buffer_dir=buffer_dir)

        src1 = tempfile.mkdtemp(prefix="low_")
        src2 = tempfile.mkdtemp(prefix="high_")
        try:
            make_fake_bin(os.path.join(src1, "low.bin"), 100)
            buf.add_games(src1, model_elo=0.0)
            make_fake_bin(os.path.join(src2, "high.bin"), 100)
            buf.add_games(src2, model_elo=200.0)

            counts = np.array([e["num_positions"] for e in buf.entries], dtype=np.float64)
            elos = np.array([e.get("model_elo", 0.0) for e in buf.entries])
            max_elo = elos.max()
            expected_scores = 1.0 / (1.0 + np.power(10.0, (max_elo - elos) / 400.0))
            # Raw weight ratio (before normalization)
            raw_weights = counts * 2.0 * expected_scores
            ratio = raw_weights[0] / raw_weights[1]
            # expected_score(200 gap) = 1/(1+10^(200/400)) = 1/(1+10^0.5) ≈ 0.240
            # expected_score(0 gap) = 1/(1+10^0) = 0.5
            # ratio = 0.240 / 0.5 ≈ 0.48
            assert 0.4 < ratio < 0.55, f"Weight ratio {ratio:.3f} should be ~0.48"
        finally:
            shutil.rmtree(src1, ignore_errors=True)
            shutil.rmtree(src2, ignore_errors=True)


class TestEpochIndices:
    def test_build_epoch_indices_same_elo_includes_all(self, tmp_dirs):
        """All entries at same Elo → 100% inclusion, all positions present."""
        buffer_dir, _ = tmp_dirs
        buf = ReplayBuffer(capacity_positions=100000, buffer_dir=buffer_dir)

        total_positions = 0
        for i in range(3):
            src = tempfile.mkdtemp(prefix=f"src{i}_")
            make_fake_bin(os.path.join(src, f"game.bin"), 50)
            buf.add_games(src, model_elo=100.0)
            total_positions += 50
            shutil.rmtree(src, ignore_errors=True)

        indices = buf.build_epoch_indices()
        assert len(indices) == total_positions  # all 150 positions included

    def test_build_epoch_indices_max_elo_fully_included(self, tmp_dirs):
        """The max-Elo entry has 100% of its positions included."""
        buffer_dir, _ = tmp_dirs
        buf = ReplayBuffer(capacity_positions=100000, buffer_dir=buffer_dir)

        src1 = tempfile.mkdtemp(prefix="low_")
        src2 = tempfile.mkdtemp(prefix="high_")
        make_fake_bin(os.path.join(src1, "low.bin"), 100)
        buf.add_games(src1, model_elo=0.0)
        make_fake_bin(os.path.join(src2, "high.bin"), 80)
        buf.add_games(src2, model_elo=200.0)
        shutil.rmtree(src1, ignore_errors=True)
        shutil.rmtree(src2, ignore_errors=True)

        indices = buf.build_epoch_indices()
        # Count positions from the max-Elo entry (entry index 1)
        max_elo_count = sum(1 for ei, _ in indices if ei == 1)
        assert max_elo_count == 80  # all positions from max-Elo entry

    def test_build_epoch_indices_weaker_model_partially_included(self, tmp_dirs):
        """200 Elo gap → ~32% inclusion (statistical test with tolerance)."""
        buffer_dir, _ = tmp_dirs
        buf = ReplayBuffer(capacity_positions=100000, buffer_dir=buffer_dir)

        src1 = tempfile.mkdtemp(prefix="low_")
        src2 = tempfile.mkdtemp(prefix="high_")
        make_fake_bin(os.path.join(src1, "low.bin"), 10000)
        buf.add_games(src1, model_elo=0.0)
        make_fake_bin(os.path.join(src2, "high.bin"), 100)
        buf.add_games(src2, model_elo=200.0)
        shutil.rmtree(src1, ignore_errors=True)
        shutil.rmtree(src2, ignore_errors=True)

        # 200 Elo gap: p = 1/(1+10^(200/400)) ≈ 0.240
        # inclusion = p/(1-p) = 0.240/0.760 ≈ 0.316
        np.random.seed(42)
        indices = buf.build_epoch_indices()
        low_elo_count = sum(1 for ei, _ in indices if ei == 0)
        inclusion_rate = low_elo_count / 10000
        assert 0.25 < inclusion_rate < 0.40, f"Inclusion rate {inclusion_rate:.3f} should be ~0.32"

    def test_build_epoch_indices_returns_valid_entry_and_position_indices(self, tmp_dirs):
        """All returned indices are in valid range."""
        buffer_dir, _ = tmp_dirs
        buf = ReplayBuffer(capacity_positions=100000, buffer_dir=buffer_dir)

        src = tempfile.mkdtemp(prefix="src_")
        make_fake_bin(os.path.join(src, "game.bin"), 50)
        buf.add_games(src, model_elo=0.0)
        shutil.rmtree(src, ignore_errors=True)

        indices = buf.build_epoch_indices()
        for entry_idx, pos_idx in indices:
            assert 0 <= entry_idx < len(buf.entries)
            assert 0 <= pos_idx < buf.entries[entry_idx]["num_positions"]

    def test_build_epoch_indices_empty_buffer_returns_empty(self, tmp_dirs):
        """Empty buffer → empty list."""
        buffer_dir, _ = tmp_dirs
        buf = ReplayBuffer(capacity_positions=100000, buffer_dir=buffer_dir)
        indices = buf.build_epoch_indices()
        assert indices == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
