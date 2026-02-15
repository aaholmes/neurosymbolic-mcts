"""Disk-backed FIFO Replay Buffer for AGZ-style training.

Manages a collection of .bin training files with a JSON manifest,
providing random sampling and FIFO eviction.
"""

import os
import json
import shutil
import time
import glob
import numpy as np

# Match train.py constants
INPUT_CHANNELS = 17
BOARD_SIZE = 64
POLICY_SIZE = 4672
SAMPLE_SIZE_FLOATS = (INPUT_CHANNELS * BOARD_SIZE) + 1 + 1 + 1 + POLICY_SIZE  # 5763 (board + material + qsearch_flag + value + policy)
BYTES_PER_SAMPLE = SAMPLE_SIZE_FLOATS * 4


class ReplayBuffer:
    def __init__(self, capacity_positions: int, buffer_dir: str):
        self.capacity = capacity_positions
        self.buffer_dir = buffer_dir
        self.entries = []  # [{path, num_positions, timestamp, model_elo}] ordered by insertion
        os.makedirs(buffer_dir, exist_ok=True)

    def add_games(self, bin_dir: str, model_elo: float = 0.0) -> int:
        """Add all .bin files from bin_dir to the buffer. Returns number of positions added.

        model_elo: cumulative Elo of the model that produced this data. Used for
        Elo-based strength weighting — data from stronger models is weighted
        higher using expected score formula.
        """
        bin_files = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
        total_added = 0

        for src_path in bin_files:
            file_size = os.path.getsize(src_path)
            if file_size == 0:
                continue
            num_positions = file_size // BYTES_PER_SAMPLE
            if num_positions == 0:
                continue

            # Copy file into buffer directory with unique name
            basename = os.path.basename(src_path)
            dst_name = f"{int(time.time() * 1000)}_{basename}"
            dst_path = os.path.join(self.buffer_dir, dst_name)
            shutil.copy2(src_path, dst_path)

            self.entries.append({
                "path": dst_path,
                "num_positions": num_positions,
                "timestamp": time.time(),
                "model_elo": model_elo,
            })
            total_added += num_positions

        self.save_manifest()
        return total_added

    def sample_batch(self, batch_size: int) -> tuple:
        """Random sample across all files. Returns (boards, scalars, values, policies) as numpy arrays.

        scalars is [B, 2] containing [material, qsearch_flag] per position.
        """
        total = self.total_positions()
        if total == 0:
            raise ValueError("Cannot sample from empty buffer")

        # Weight files by position count, scaled by Elo-based expected score.
        # Data from stronger models (higher Elo) is weighted more heavily.
        # expected_score = 1 / (1 + 10^((max_elo - entry_elo) / 400))
        # weight = num_positions * 2 * expected_score  (so best model → weight 1.0 per position)
        counts = np.array([e["num_positions"] for e in self.entries], dtype=np.float64)
        elos = np.array([e.get("model_elo", 0.0) for e in self.entries])
        max_elo = elos.max() if len(elos) > 0 else 0.0
        expected_scores = 1.0 / (1.0 + np.power(10.0, (max_elo - elos) / 400.0))
        weights = counts * 2.0 * expected_scores

        weights /= weights.sum()

        boards = np.empty((batch_size, INPUT_CHANNELS, 8, 8), dtype=np.float32)
        scalars = np.empty((batch_size, 2), dtype=np.float32)
        values = np.empty((batch_size, 1), dtype=np.float32)
        policies = np.empty((batch_size, POLICY_SIZE), dtype=np.float32)

        # Pick random file indices weighted by position count
        file_indices = np.random.choice(len(self.entries), size=batch_size, p=weights)

        for i, fi in enumerate(file_indices):
            entry = self.entries[fi]
            # Pick random position within file
            pos_idx = np.random.randint(0, entry["num_positions"])
            offset = pos_idx * BYTES_PER_SAMPLE

            with open(entry["path"], "rb") as f:
                f.seek(offset)
                raw = np.frombuffer(f.read(BYTES_PER_SAMPLE), dtype=np.float32)

            board_end = INPUT_CHANNELS * BOARD_SIZE
            boards[i] = raw[:board_end].reshape(INPUT_CHANNELS, 8, 8)
            scalars[i, 0] = raw[board_end]       # material
            scalars[i, 1] = raw[board_end + 1]   # qsearch_flag
            values[i, 0] = raw[board_end + 2]
            policies[i] = raw[board_end + 3:]

        return boards, scalars, values, policies

    def build_epoch_indices(self):
        """Build indices for one epoch with Elo-weighted probabilistic inclusion.

        Returns list of (entry_idx, pos_idx) tuples.
        - Max-Elo entries: 100% inclusion (all positions)
        - Older entries: inclusion_prob = min(1.0, p/(1-p))
          where p = 1/(1+10^((max_elo-elo)/400))
        """
        if not self.entries:
            return []

        elos = np.array([e.get("model_elo", 0.0) for e in self.entries])
        max_elo = elos.max()

        indices = []
        for entry_idx, entry in enumerate(self.entries):
            n = entry["num_positions"]
            elo = entry.get("model_elo", 0.0)
            delta = max_elo - elo

            if delta <= 0:
                # Max Elo: include all positions
                indices.extend((entry_idx, pos_idx) for pos_idx in range(n))
            else:
                # p = expected score, inclusion = p/(1-p) = odds ratio
                p = 1.0 / (1.0 + 10.0 ** (delta / 400.0))
                inclusion = min(1.0, p / (1.0 - p))
                if inclusion >= 1.0:
                    indices.extend((entry_idx, pos_idx) for pos_idx in range(n))
                else:
                    mask = np.random.random(n) < inclusion
                    for pos_idx in np.where(mask)[0]:
                        indices.append((entry_idx, int(pos_idx)))

        return indices

    def evict_oldest(self):
        """Remove oldest files until total positions <= capacity."""
        while self.total_positions() > self.capacity and self.entries:
            oldest = self.entries.pop(0)
            try:
                os.remove(oldest["path"])
            except FileNotFoundError:
                pass
        self.save_manifest()

    def total_positions(self) -> int:
        return sum(e["num_positions"] for e in self.entries)

    def save_manifest(self):
        manifest_path = os.path.join(self.buffer_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(self.entries, f, indent=2)

    def load_manifest(self):
        manifest_path = os.path.join(self.buffer_dir, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r") as f:
                self.entries = json.load(f)
            # Filter out entries whose files no longer exist
            self.entries = [e for e in self.entries if os.path.exists(e["path"])]
