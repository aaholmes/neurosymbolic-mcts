import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import struct
import glob
from model import CaissawaryNet

# Configuration
INPUT_CHANNELS = 12
BOARD_SIZE = 8 * 8
POLICY_SIZE = 4096
SAMPLE_SIZE_FLOATS = (INPUT_CHANNELS * BOARD_SIZE) + 1 + 1 + POLICY_SIZE 
# 768 (Board) + 1 (Material) + 1 (Value) + 4096 (Policy) = 4866 floats

class ChessDataset(Dataset):
    def __init__(self, data_dir):
        self.files = glob.glob(os.path.join(data_dir, "*.bin"))
        self.samples = []
        
        print(f"Loading data from {len(self.files)} files...")
        for file_path in self.files:
            self._load_file(file_path)
        print(f"Loaded {len(self.samples)} training samples.")

    def _load_file(self, path):
        # Determine number of samples based on file size
        file_size = os.path.getsize(path)
        bytes_per_sample = SAMPLE_SIZE_FLOATS * 4 # 4 bytes per float32
        num_samples = file_size // bytes_per_sample
        
        with open(path, 'rb') as f:
            data = np.fromfile(f, dtype=np.float32)
            
        if data.size == 0:
            return

        # Reshape to [N, SAMPLE_SIZE_FLOATS]
        try:
            data = data.reshape(num_samples, SAMPLE_SIZE_FLOATS)
            
            # Split into components (using views for memory efficiency)
            # 1. Board: First 768 floats
            board_end = INPUT_CHANNELS * BOARD_SIZE
            self.samples.extend(data) 
            # Note: storing the full flat row to avoid memory fragmentation per sample
            # We'll slice in __getitem__
        except ValueError as e:
            print(f"Error loading {path}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        flat_data = self.samples[idx]
        
        # 1. Board [12, 8, 8]
        board_end = INPUT_CHANNELS * BOARD_SIZE
        board_data = flat_data[:board_end]
        board_tensor = torch.from_numpy(board_data).view(INPUT_CHANNELS, 8, 8)
        
        # 2. Material Scalar [1]
        material_idx = board_end
        material_scalar = torch.tensor([flat_data[material_idx]])
        
        # 3. Value Target [1]
        value_idx = material_idx + 1
        value_target = torch.tensor([flat_data[value_idx]])
        
        # 4. Policy Target [4096]
        policy_start = value_idx + 1
        policy_target = torch.from_numpy(flat_data[policy_start:])
        
        return board_tensor, material_scalar, value_target, policy_target

def train():
    # Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 10
    DATA_DIR = "data/training" # Ensure this matches Rust output
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # Data
    dataset = ChessDataset(DATA_DIR)
    if len(dataset) == 0:
        print("No training data found. Run self-play first.")
        return
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model
    model = CaissawaryNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Training Loop
    model.train()
    for epoch in range(EPOCHS):
        total_policy_loss = 0
        total_value_loss = 0
        
        for batch_idx, (boards, materials, values, policies) in enumerate(dataloader):
            boards = boards.to(DEVICE)
            materials = materials.to(DEVICE)
            values = values.to(DEVICE)
            policies = policies.to(DEVICE)
            
            # Forward
            pred_policy, pred_value = model(boards, materials)
            
            # Loss
            # Policy: KLDiv expects log_probs (pred) and probs (target)
            # pred_policy is already LogSoftmax from model
            policy_loss = F.kl_div(pred_policy, policies, reduction='batchmean')
            
            # Value: MSE
            value_loss = F.mse_loss(pred_value, values)
            
            loss = policy_loss + value_loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} Batch {batch_idx}: P_Loss={policy_loss.item():.4f} V_Loss={value_loss.item():.4f}")

        print(f"Epoch {epoch+1} Average: Policy={total_policy_loss/len(dataloader):.4f} Value={total_value_loss/len(dataloader):.4f}")
        
        # Save checkpoint
        os.makedirs("python/models", exist_ok=True)
        torch.save(model.state_dict(), "python/models/latest.pt")
        print("Saved python/models/latest.pt")

if __name__ == "__main__":
    import torch.nn.functional as F
    train()
