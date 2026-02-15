import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import os
import struct
import glob
import sys
from model import OracleNet
from augmentation import augment_sample, augment_all_transforms

# Configuration
INPUT_CHANNELS = 17
BOARD_SIZE = 8 * 8
POLICY_SIZE = 4672
SAMPLE_SIZE_FLOATS = (INPUT_CHANNELS * BOARD_SIZE) + 1 + 1 + 1 + POLICY_SIZE  # board + material + qsearch_flag + value + policy

class ChessDataset(Dataset):
    def __init__(self, data_dir, augment=False):
        self.files = glob.glob(os.path.join(data_dir, "*.bin"))
        self.samples = []
        self.augment = augment
        self._rng = np.random.default_rng()

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
        board_np = board_data.reshape(INPUT_CHANNELS, 8, 8)

        # 2. Material Scalar [1]
        material_idx = board_end
        material = flat_data[material_idx]

        # 3. Q-search Completed Flag [1]
        qsearch_idx = material_idx + 1
        qsearch_flag = flat_data[qsearch_idx]

        # 4. Value Target [1]
        value_idx = qsearch_idx + 1
        value = flat_data[value_idx]

        # 5. Policy Target [4672]
        policy_start = value_idx + 1
        policy_np = flat_data[policy_start:]

        # 6. Augmentation (randomly pick one transform, weight 1.0)
        if self.augment:
            transforms = augment_all_transforms(board_np, material, value, policy_np)
            idx = self._rng.integers(len(transforms))
            board_np, material, value, policy_np = transforms[idx]

        board_tensor = torch.from_numpy(np.ascontiguousarray(board_np))
        scalars = torch.tensor([material, qsearch_flag])  # [2]
        value_target = torch.tensor([value])
        policy_target = torch.from_numpy(np.ascontiguousarray(policy_np))

        return board_tensor, scalars, value_target, policy_target


class BufferDataset(Dataset):
    """Dataset that samples from a replay buffer directory.

    Pre-samples chunks into memory to avoid per-item file opens.
    """

    def __init__(self, buffer_dir, chunk_size=4096, augment=False):
        from replay_buffer import ReplayBuffer
        self.buffer = ReplayBuffer(capacity_positions=10**9, buffer_dir=buffer_dir)
        self.buffer.load_manifest()
        self._total = self.buffer.total_positions()
        self.chunk_size = chunk_size
        self._chunk = None
        self._chunk_idx = 0
        self.augment = augment
        self._rng = np.random.default_rng()
        self._refresh_chunk()
        print(f"Buffer dataset: {self._total} positions from {len(self.buffer.entries)} files")

    def _refresh_chunk(self):
        boards, scalars, values, policies = self.buffer.sample_batch(self.chunk_size)
        if self.augment:
            aug_boards, aug_scalars, aug_vals, aug_pols = [], [], [], []
            for i in range(len(boards)):
                for b, m, v, p in augment_all_transforms(
                    boards[i], scalars[i, 0], values[i], policies[i]
                ):
                    aug_boards.append(b)
                    aug_scalars.append([m, scalars[i, 1]])  # preserve qsearch_flag
                    aug_vals.append(v)
                    aug_pols.append(p)
            self._chunk = (
                np.array(aug_boards),
                np.array(aug_scalars),
                np.array(aug_vals),
                np.array(aug_pols),
            )
        else:
            self._chunk = (boards, scalars, values, policies)
        self._chunk_len = len(self._chunk[0])
        self._chunk_idx = 0

    def __len__(self):
        return self._total

    def __getitem__(self, idx):
        if self._chunk_idx >= self._chunk_len:
            self._refresh_chunk()
        i = self._chunk_idx
        self._chunk_idx += 1

        board_np = self._chunk[0][i]
        scalars_np = self._chunk[1][i]  # [2] array: [material, qsearch_flag]
        value = self._chunk[2][i]
        policy_np = self._chunk[3][i]

        return (torch.from_numpy(np.ascontiguousarray(board_np)),
                torch.from_numpy(np.ascontiguousarray(scalars_np).astype(np.float32)),
                torch.tensor([value.item()]) if hasattr(value, 'item') else torch.tensor([value]),
                torch.from_numpy(np.ascontiguousarray(policy_np)))


class EpochDataset(Dataset):
    """Dataset for epoch-based training with Elo-weighted inclusion.

    Uses build_epoch_indices() to probabilistically include positions
    based on the Elo of the model that produced them. Reads positions
    from disk on demand (one file open per __getitem__).
    """

    def __init__(self, buffer_dir, augment=False):
        from replay_buffer import ReplayBuffer, BYTES_PER_SAMPLE as BPS, INPUT_CHANNELS as IC, BOARD_SIZE as BS, POLICY_SIZE as PS
        self.buffer = ReplayBuffer(capacity_positions=10**9, buffer_dir=buffer_dir)
        self.buffer.load_manifest()
        self._bps = BPS
        self._ic = IC
        self._bs = BS
        self._ps = PS
        self._board_end = IC * BS
        self.augment = augment
        self._rng = np.random.default_rng()
        self.indices = self.buffer.build_epoch_indices()
        total = self.buffer.total_positions()
        print(f"Epoch dataset: {len(self.indices)} positions included from {total} total "
              f"({len(self.buffer.entries)} files)")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        entry_idx, pos_idx = self.indices[idx]
        entry = self.buffer.entries[entry_idx]
        offset = pos_idx * self._bps

        with open(entry["path"], "rb") as f:
            f.seek(offset)
            raw = np.frombuffer(f.read(self._bps), dtype=np.float32)

        board_np = raw[:self._board_end].reshape(self._ic, 8, 8)
        material = raw[self._board_end]
        qsearch_flag = raw[self._board_end + 1]
        value = raw[self._board_end + 2]
        policy_np = raw[self._board_end + 3:]

        if self.augment:
            transforms = augment_all_transforms(board_np, material, value, policy_np)
            t_idx = self._rng.integers(len(transforms))
            board_np, material, value, policy_np = transforms[t_idx]

        board_tensor = torch.from_numpy(np.ascontiguousarray(board_np))
        scalars = torch.tensor([material, qsearch_flag])  # [2]
        value_target = torch.tensor([value])
        policy_target = torch.from_numpy(np.ascontiguousarray(policy_np))

        return board_tensor, scalars, value_target, policy_target


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


def freeze_heads(model, train_heads):
    """Freeze parameters not in the specified head group.

    train_heads: 'all' (default), 'policy' (freeze value+k), 'value' (freeze policy).
    The backbone is always trained.
    """
    if train_heads == "all":
        return

    # Parameter name prefixes for each head
    policy_prefixes = ("p_conv", "p_bn", "p_head")
    value_prefixes = ("v_conv", "v_bn", "v_fc", "v_out")
    k_prefixes = ("k_stm_patch_fc", "k_opp_patch_fc", "k_combine", "k_out")

    if train_heads == "policy":
        frozen = value_prefixes + k_prefixes
    elif train_heads == "value":
        frozen = policy_prefixes
    else:
        raise ValueError(f"Unknown train_heads={train_heads!r}, expected 'all', 'policy', or 'value'")

    count = 0
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in frozen):
            param.requires_grad = False
            count += 1

    print(f"Frozen {count} parameters (train_heads={train_heads})")


def make_optimizer(model, optimizer_name, lr):
    """Create optimizer by name. Only includes parameters with requires_grad."""
    params = [p for p in model.parameters() if p.requires_grad]
    if optimizer_name == 'muon':
        from muon import Muon
        return Muon(params, lr=lr, backend_lr=lr * 0.1)
    elif optimizer_name == 'adamw':
        return optim.AdamW(params, lr=lr, weight_decay=1e-4)
    else:
        return optim.Adam(params, lr=lr, weight_decay=1e-4)


def train_with_config(
    data_dir,
    output_path,
    resume_path=None,
    optimizer_name="adam",
    lr=0.001,
    max_epochs=2,
    batch_size=64,
    minibatches=None,
    lr_schedule="",
    buffer_dir=None,
    _return_lr_history=False,
    disable_material=False,
    augment=True,
    reset_optimizer=False,
    train_heads="all",
    num_blocks=6,
    hidden_dim=128,
    use_epochs=False,
    # Backward compat alias
    epochs=None,
):
    """Core training function. Returns number of minibatches trained.

    When _return_lr_history=True, returns (step_count, lr_history_dict) for testing.
    """
    # Handle backward compat: if caller passes epochs= but not max_epochs=, use it
    if epochs is not None:
        max_epochs = epochs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    if use_epochs and buffer_dir:
        dataset = EpochDataset(buffer_dir, augment=augment)
    elif buffer_dir:
        dataset = BufferDataset(buffer_dir, augment=augment)
    else:
        dataset = ChessDataset(data_dir, augment=augment)

    if len(dataset) == 0:
        print("No training data found.")
        return 0

    # For epoch mode with max_epochs > 1, split into train/val
    val_loader = None
    if use_epochs and max_epochs > 1 and len(dataset) >= 20:
        n = len(dataset)
        n_val = max(1, int(n * 0.1))
        n_train = n - n_val
        rng = torch.Generator().manual_seed(42)
        indices = torch.randperm(n, generator=rng).tolist()
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
            num_workers=2, pin_memory=True, persistent_workers=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
            num_workers=0, pin_memory=True,
        )
        print(f"Train/val split: {n_train} train, {n_val} val")
    else:
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=False,
            num_workers=2, pin_memory=True, persistent_workers=True,
        )

    # Model
    model = OracleNet(num_blocks=num_blocks, hidden_dim=hidden_dim).to(DEVICE)

    # Resume from checkpoint
    global_minibatch = 0
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}")
        try:
            checkpoint = torch.load(resume_path, map_location=DEVICE, weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                global_minibatch = checkpoint.get("global_minibatch", 0)
            else:
                # Legacy format: just state_dict
                model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Warning: Failed to load resume checkpoint: {e}")

    # Freeze heads BEFORE creating optimizer (so optimizer only tracks trainable params)
    freeze_heads(model, train_heads)
    optimizer = make_optimizer(model, optimizer_name, lr)

    if resume_path and os.path.exists(resume_path) and not reset_optimizer:
        try:
            checkpoint = torch.load(resume_path, map_location=DEVICE, weights_only=False)
            if isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                except Exception:
                    pass  # Optimizer mismatch is OK (e.g., different param count after freeze)
        except Exception:
            pass

    # LR schedule
    schedule = parse_lr_schedule(lr_schedule)

    # Training Loop
    model.train()
    steps_this_run = 0
    lr_history = {}

    # use_epochs forces epoch-based training (ignoring minibatches)
    if use_epochs:
        minibatches = None

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

            boards, scalars, values, policies = batch
            boards = boards.to(DEVICE)
            scalars = scalars.to(DEVICE)
            if disable_material:
                scalars = scalars.clone()
                scalars[:, 0] = 0.0  # zero out material column only
            values = values.to(DEVICE)
            policies = policies.to(DEVICE)

            pred_policy, pred_value, k = model(boards, scalars)
            policy_loss_per = F.kl_div(pred_policy, policies, reduction='none').sum(dim=1)
            value_loss_per = F.mse_loss(pred_value, values, reduction='none').squeeze(1)
            loss = (policy_loss_per + value_loss_per).mean()
            policy_loss = policy_loss_per.mean()
            value_loss = value_loss_per.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_minibatch += 1
            steps_this_run += 1

            # Loss logging every 100 minibatches
            if steps_this_run % 100 == 0:
                print(f"  Step {steps_this_run}/{minibatches} (global {global_minibatch}): "
                      f"Loss={loss.item():.4f} P={policy_loss.item():.4f} "
                      f"V={value_loss.item():.4f} K={k.mean().item():.4f} LR={current_lr}")

            # Mid-training checkpoint every 200 minibatches
            if steps_this_run > 0 and steps_this_run % 200 == 0:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_minibatch": global_minibatch,
                }, output_path)
    else:
        # Epoch mode with optional early stopping via validation loss
        best_val_loss = float('inf')
        best_epoch = 0
        best_state = None
        patience = 1  # stop after 1 epoch of no improvement

        for epoch in range(max_epochs):
            # --- Train ---
            model.train()
            total_policy_loss = 0
            total_value_loss = 0
            total_k = 0

            for batch_idx, (boards, scalars, values, policies) in enumerate(dataloader):
                # Apply LR schedule
                current_lr = get_lr_for_step(schedule, lr, global_minibatch)
                for pg in optimizer.param_groups:
                    pg['lr'] = current_lr

                if _return_lr_history:
                    lr_history[steps_this_run] = current_lr

                boards = boards.to(DEVICE)
                scalars = scalars.to(DEVICE)
                if disable_material:
                    scalars = scalars.clone()
                    scalars[:, 0] = 0.0  # zero out material column only
                values = values.to(DEVICE)
                policies = policies.to(DEVICE)

                pred_policy, pred_value, k = model(boards, scalars)
                policy_loss_per = F.kl_div(pred_policy, policies, reduction='none').sum(dim=1)
                value_loss_per = F.mse_loss(pred_value, values, reduction='none').squeeze(1)
                loss = (policy_loss_per + value_loss_per).mean()
                policy_loss = policy_loss_per.mean()
                value_loss = value_loss_per.mean()

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

            # --- Validation ---
            if val_loader is not None:
                model.eval()
                val_policy_total = 0
                val_value_total = 0
                val_batches = 0
                with torch.no_grad():
                    for boards, scalars, values, policies in val_loader:
                        boards = boards.to(DEVICE)
                        scalars = scalars.to(DEVICE)
                        if disable_material:
                            scalars = scalars.clone()
                            scalars[:, 0] = 0.0
                        values = values.to(DEVICE)
                        policies = policies.to(DEVICE)

                        pred_policy, pred_value, k_val = model(boards, scalars)
                        p_loss = F.kl_div(pred_policy, policies, reduction='none').sum(dim=1).mean()
                        v_loss = F.mse_loss(pred_value, values)
                        val_policy_total += p_loss.item()
                        val_value_total += v_loss.item()
                        val_batches += 1

                if val_batches > 0:
                    val_p = val_policy_total / val_batches
                    val_v = val_value_total / val_batches
                    val_loss_avg = val_p + val_v
                    if n_batches > 0:
                        print(f"Epoch {epoch+1} Train: P={avg_policy:.4f} V={avg_value:.4f} | "
                              f"Val: P={val_p:.4f} V={val_v:.4f} (total={val_loss_avg:.4f})")
                    else:
                        print(f"Epoch {epoch+1} Val: P={val_p:.4f} V={val_v:.4f} (total={val_loss_avg:.4f})")

                    # Handle NaN: treat as no improvement but don't early stop
                    if math.isnan(val_loss_avg):
                        if best_state is None:
                            best_epoch = epoch + 1
                            best_state = copy.deepcopy(model.state_dict())
                    elif val_loss_avg < best_val_loss:
                        best_val_loss = val_loss_avg
                        best_epoch = epoch + 1
                        best_state = copy.deepcopy(model.state_dict())
                    elif (epoch + 1) - best_epoch >= patience:
                        print(f"Early stopping at epoch {epoch+1} (best was epoch {best_epoch})")
                        break
            else:
                # No validation — just log train averages
                if n_batches > 0:
                    print(f"Epoch {epoch+1} Average: Policy={avg_policy:.4f} Value={avg_value:.4f} K={avg_k:.4f}")

        # Restore best model if we did validation
        if best_state is not None:
            model.load_state_dict(best_state)
            print(f"BEST_EPOCH={best_epoch}")
            print(f"VAL_LOSS={best_val_loss:.6f}")

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
    parser = argparse.ArgumentParser(description="Train OracleNet chess model")
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
    parser.add_argument('--max-epochs', type=int, default=10, dest='max_epochs',
                        help='Maximum training epochs with early stopping (default: 10)')
    parser.add_argument('--epochs', type=int, default=None, dest='epochs_alias',
                        help=argparse.SUPPRESS)  # hidden backward compat alias
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--minibatches', type=int, default=None,
                        help='Number of minibatches to train (overrides --epochs when set)')
    parser.add_argument('--lr-schedule', type=str, default='',
                        help='LR schedule: "step:lr,step:lr" (e.g., "500000:0.01,1000000:0.001")')
    parser.add_argument('--buffer-dir', type=str, default=None,
                        help='Replay buffer directory (overrides data_dir when set)')
    parser.add_argument('--disable-material', action='store_true',
                        help='Zero out material scalars (for pure AlphaZero baseline)')
    parser.add_argument('--reset-optimizer', action='store_true',
                        help='Reset optimizer state (fresh momentum after buffer clear)')
    parser.add_argument('--augment', action='store_true', default=True,
                        help='Enable symmetry augmentation (default: enabled)')
    parser.add_argument('--no-augment', action='store_false', dest='augment',
                        help='Disable symmetry augmentation')
    parser.add_argument('--train-heads', type=str, default='all',
                        choices=['all', 'policy', 'value'],
                        help='Which heads to train: all, policy (freeze value+k), value (freeze policy)')
    parser.add_argument('--num-blocks', type=int, default=6,
                        help='Number of residual blocks in OracleNet (default: 6)')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension of OracleNet (default: 128)')
    parser.add_argument('--use-epochs', action='store_true',
                        help='Epoch-based training with Elo-weighted inclusion (ignores --minibatches)')
    return parser.parse_args()


def train():
    args = parse_args()

    # Resolve max_epochs: --epochs alias overrides --max-epochs if provided
    max_epochs = args.epochs_alias if args.epochs_alias is not None else args.max_epochs

    # Set default LR based on optimizer
    if args.lr is None:
        lr = 0.02 if args.optimizer == 'muon' else 0.001
    else:
        lr = args.lr

    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Data Dir: {args.data_dir}")
    print(f"Output Model: {args.output_path}")
    print(f"Optimizer: {args.optimizer} (lr={lr})")
    if args.use_epochs:
        print(f"Epoch-based training: up to {max_epochs} epoch(s)")
    elif args.minibatches:
        print(f"Minibatches: {args.minibatches}")
    else:
        print(f"Epochs: {max_epochs}")
    if args.lr_schedule:
        print(f"LR Schedule: {args.lr_schedule}")
    if args.buffer_dir:
        print(f"Buffer Dir: {args.buffer_dir}")
    print(f"Augmentation: {'enabled' if args.augment else 'disabled'}")
    train_with_config(
        data_dir=args.data_dir,
        output_path=args.output_path,
        resume_path=args.resume_path,
        optimizer_name=args.optimizer,
        lr=lr,
        max_epochs=max_epochs,
        batch_size=args.batch_size,
        minibatches=args.minibatches,
        lr_schedule=args.lr_schedule,
        buffer_dir=args.buffer_dir,
        disable_material=args.disable_material,
        augment=args.augment,
        reset_optimizer=args.reset_optimizer,
        train_heads=args.train_heads,
        num_blocks=args.num_blocks,
        hidden_dim=args.hidden_dim,
        use_epochs=args.use_epochs,
    )


if __name__ == "__main__":
    train()
