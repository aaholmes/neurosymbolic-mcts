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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
