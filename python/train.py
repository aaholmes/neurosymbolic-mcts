import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import struct
import glob
import sys
from model import LogosNet

# Configuration
INPUT_CHANNELS = 17
BOARD_SIZE = 8 * 8
POLICY_SIZE = 4672
SAMPLE_SIZE_FLOATS = (INPUT_CHANNELS * BOARD_SIZE) + 1 + 1 + POLICY_SIZE

class ChessDataset(Dataset):
    def __init__(self, data_dir):
        self.files = glob.glob(os.path.join(data_dir, "*.bin"))
        self.samples = []

        print(f"Loading data from {len(self.files)} files in {data_dir}...")
        for file_path in self.files:
            self._load_file(file_path)
        print(f"Loaded {len(self.samples)} training samples.")

    def _load_file(self, path):
        file_size = os.path.getsize(path)
        bytes_per_sample = SAMPLE_SIZE_FLOATS * 4
        num_samples = file_size // bytes_per_sample

        with open(path, 'rb') as f:
            data = np.fromfile(f, dtype=np.float32)

        if data.size == 0:
            return

        try:
            data = data.reshape(num_samples, SAMPLE_SIZE_FLOATS)
            board_end = INPUT_CHANNELS * BOARD_SIZE
            self.samples.extend(data)
        except ValueError as e:
            print(f"Error loading {path}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        flat_data = self.samples[idx]

        # 1. Board [17, 8, 8]
        board_end = INPUT_CHANNELS * BOARD_SIZE
        board_data = flat_data[:board_end]
        board_tensor = torch.from_numpy(board_data).view(INPUT_CHANNELS, 8, 8)

        # 2. Material Scalar [1]
        material_idx = board_end
        material_scalar = torch.tensor([flat_data[material_idx]])

        # 3. Value Target [1]
        value_idx = material_idx + 1
        value_target = torch.tensor([flat_data[value_idx]])

        # 4. Policy Target [4672]
        policy_start = value_idx + 1
        policy_target = torch.from_numpy(flat_data[policy_start:])

        return board_tensor, material_scalar, value_target, policy_target


class BufferDataset(Dataset):
    """Dataset that samples from a replay buffer directory."""

    def __init__(self, buffer_dir):
        from replay_buffer import ReplayBuffer
        self.buffer = ReplayBuffer(capacity_positions=10**9, buffer_dir=buffer_dir)
        self.buffer.load_manifest()
        self._total = self.buffer.total_positions()
        print(f"Buffer dataset: {self._total} positions from {len(self.buffer.entries)} files")

    def __len__(self):
        return self._total

    def __getitem__(self, idx):
        # Sample one random position from buffer (ignore idx, always random)
        boards, materials, values, policies = self.buffer.sample_batch(1)
        board_tensor = torch.from_numpy(boards[0])
        material_scalar = torch.tensor(materials[0])
        value_target = torch.tensor(values[0])
        policy_target = torch.from_numpy(policies[0])
        return board_tensor, material_scalar, value_target, policy_target


def parse_lr_schedule(schedule_str):
    """Parse LR schedule string like '500:0.01,1000:0.001' into sorted list of (step, lr) tuples."""
    if not schedule_str:
        return []
    pairs = []
    for part in schedule_str.split(","):
        step_str, lr_str = part.strip().split(":")
        pairs.append((int(step_str), float(lr_str)))
    return sorted(pairs, key=lambda x: x[0])


def get_lr_for_step(schedule, base_lr, step):
    """Get the learning rate for a given global step based on the schedule."""
    current_lr = base_lr
    for boundary, lr in schedule:
        if step >= boundary:
            current_lr = lr
        else:
            break
    return current_lr


def make_optimizer(model, optimizer_name, lr):
    """Create optimizer by name."""
    if optimizer_name == 'muon':
        from muon import Muon
        return Muon(model.parameters(), lr=lr, backend_lr=lr * 0.1)
    elif optimizer_name == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        return optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)


def train_with_config(
    data_dir,
    output_path,
    resume_path=None,
    optimizer_name="adam",
    lr=0.001,
    epochs=2,
    batch_size=64,
    minibatches=None,
    lr_schedule="",
    buffer_dir=None,
    _return_lr_history=False,
):
    """Core training function. Returns number of minibatches trained.

    When _return_lr_history=True, returns (step_count, lr_history_dict) for testing.
    """
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    if buffer_dir:
        dataset = BufferDataset(buffer_dir)
    else:
        dataset = ChessDataset(data_dir)

    if len(dataset) == 0:
        print("No training data found.")
        return 0

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # Model
    model = LogosNet().to(DEVICE)
    optimizer = make_optimizer(model, optimizer_name, lr)

    # Resume from checkpoint
    global_minibatch = 0
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}")
        try:
            checkpoint = torch.load(resume_path, map_location=DEVICE, weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer = make_optimizer(model, optimizer_name, lr)
                if "optimizer_state_dict" in checkpoint:
                    try:
                        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    except Exception:
                        pass  # Optimizer mismatch is OK, just use fresh
                global_minibatch = checkpoint.get("global_minibatch", 0)
            else:
                # Legacy format: just state_dict
                model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Warning: Failed to load resume checkpoint: {e}")

    # LR schedule
    schedule = parse_lr_schedule(lr_schedule)

    # Training Loop
    model.train()
    steps_this_run = 0
    lr_history = {}

    if minibatches is not None:
        # Minibatch-count mode: run exactly N minibatches
        data_iter = iter(dataloader)
        for _ in range(minibatches):
            # Apply LR schedule
            current_lr = get_lr_for_step(schedule, lr, global_minibatch)
            for pg in optimizer.param_groups:
                pg['lr'] = current_lr

            if _return_lr_history:
                lr_history[steps_this_run] = current_lr

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            boards, materials, values, policies = batch
            boards = boards.to(DEVICE)
            materials = materials.to(DEVICE)
            values = values.to(DEVICE)
            policies = policies.to(DEVICE)

            pred_policy, pred_value, k = model(boards, materials)
            policy_loss = F.kl_div(pred_policy, policies, reduction='batchmean')
            value_loss = F.mse_loss(pred_value, values)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_minibatch += 1
            steps_this_run += 1
    else:
        # Epoch mode (original behavior)
        for epoch in range(epochs):
            total_policy_loss = 0
            total_value_loss = 0
            total_k = 0

            for batch_idx, (boards, materials, values, policies) in enumerate(dataloader):
                # Apply LR schedule
                current_lr = get_lr_for_step(schedule, lr, global_minibatch)
                for pg in optimizer.param_groups:
                    pg['lr'] = current_lr

                if _return_lr_history:
                    lr_history[steps_this_run] = current_lr

                boards = boards.to(DEVICE)
                materials = materials.to(DEVICE)
                values = values.to(DEVICE)
                policies = policies.to(DEVICE)

                pred_policy, pred_value, k = model(boards, materials)
                policy_loss = F.kl_div(pred_policy, policies, reduction='batchmean')
                value_loss = F.mse_loss(pred_value, values)
                loss = policy_loss + value_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                current_k = k.mean().item()
                total_k += current_k

                global_minibatch += 1
                steps_this_run += 1

                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1} Batch {batch_idx}: Loss={loss.item():.4f} "
                          f"P_Loss={policy_loss.item():.4f} V_Loss={value_loss.item():.4f} K={current_k:.4f}")

            n_batches = len(dataloader)
            if n_batches > 0:
                avg_policy = total_policy_loss / n_batches
                avg_value = total_value_loss / n_batches
                avg_k = total_k / n_batches
                print(f"Epoch {epoch+1} Average: Policy={avg_policy:.4f} Value={avg_value:.4f} K={avg_k:.4f}")

    # Save full checkpoint
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_minibatch": global_minibatch,
    }, output_path)
    print(f"Saved checkpoint to {output_path} (global_minibatch={global_minibatch})")

    if _return_lr_history:
        return steps_this_run, lr_history
    return steps_this_run


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Train LogosNet chess model")
    parser.add_argument('data_dir', type=str, nargs='?', default='data/training',
                        help='Directory containing .bin training data')
    parser.add_argument('output_path', type=str, nargs='?', default='python/models/latest.pt',
                        help='Output model path')
    parser.add_argument('resume_path', type=str, nargs='?', default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adamw', 'muon'],
                        help='Optimizer to use (default: adam)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: 0.001 for adam/adamw, 0.02 for muon)')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of training epochs (default: 2)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--minibatches', type=int, default=None,
                        help='Number of minibatches to train (overrides --epochs when set)')
    parser.add_argument('--lr-schedule', type=str, default='',
                        help='LR schedule: "step:lr,step:lr" (e.g., "500000:0.01,1000000:0.001")')
    parser.add_argument('--buffer-dir', type=str, default=None,
                        help='Replay buffer directory (overrides data_dir when set)')
    return parser.parse_args()


def train():
    args = parse_args()

    # Set default LR based on optimizer
    if args.lr is None:
        lr = 0.02 if args.optimizer == 'muon' else 0.001
    else:
        lr = args.lr

    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Data Dir: {args.data_dir}")
    print(f"Output Model: {args.output_path}")
    print(f"Optimizer: {args.optimizer} (lr={lr})")
    if args.minibatches:
        print(f"Minibatches: {args.minibatches}")
    else:
        print(f"Epochs: {args.epochs}")
    if args.lr_schedule:
        print(f"LR Schedule: {args.lr_schedule}")
    if args.buffer_dir:
        print(f"Buffer Dir: {args.buffer_dir}")

    train_with_config(
        data_dir=args.data_dir,
        output_path=args.output_path,
        resume_path=args.resume_path,
        optimizer_name=args.optimizer,
        lr=lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        minibatches=args.minibatches,
        lr_schedule=args.lr_schedule,
        buffer_dir=args.buffer_dir,
    )


if __name__ == "__main__":
    train()
