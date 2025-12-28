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
        
        # 4. Policy Target [4096]
        policy_start = value_idx + 1
        policy_target = torch.from_numpy(flat_data[policy_start:])
        
        return board_tensor, material_scalar, value_target, policy_target

def train():
    # Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Command line args
    if len(sys.argv) < 3:
        print("Usage: python train.py <data_dir> <output_model_path> [resume_path]")
        DATA_DIR = "data/training"
        OUTPUT_PATH = "python/models/latest.pt"
    else:
        DATA_DIR = sys.argv[1]
        OUTPUT_PATH = sys.argv[2]

    print(f"Using device: {DEVICE}")
    print(f"Data Dir: {DATA_DIR}")
    print(f"Output Model: {OUTPUT_PATH}")

    # Data
    dataset = ChessDataset(DATA_DIR)
    if len(dataset) == 0:
        print("No training data found.")
        return
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model
    model = LogosNet().to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    if len(sys.argv) >= 4:
        resume_path = sys.argv[3]
        if os.path.exists(resume_path):
            print(f"Resuming from {resume_path}")
            try:
                model.load_state_dict(torch.load(resume_path, map_location=DEVICE))
            except Exception as e:
                print(f"Warning: Failed to load resume checkpoint: {e}")

    # Training Loop
    model.train()
    for epoch in range(EPOCHS):
        total_policy_loss = 0
        total_value_loss = 0
        total_k = 0
        
        for batch_idx, (boards, materials, values, policies) in enumerate(dataloader):
            boards = boards.to(DEVICE)
            materials = materials.to(DEVICE)
            values = values.to(DEVICE)
            policies = policies.to(DEVICE)
            
            # Forward: returns (policy, value, k)
            pred_policy, pred_value, k = model(boards, materials)
            
            # Loss
            policy_loss = F.kl_div(pred_policy, policies, reduction='batchmean')
            value_loss = F.mse_loss(pred_value, values)
            
            loss = policy_loss + value_loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            current_k = k.mean().item()
            total_k += current_k
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} Batch {batch_idx}: P_Loss={policy_loss.item():.4f} V_Loss={value_loss.item():.4f} K={current_k:.4f}")

        avg_k = total_k / len(dataloader)
        print(f"Epoch {epoch+1} Average: Policy={total_policy_loss/len(dataloader):.4f} Value={total_value_loss/len(dataloader):.4f} K={avg_k:.4f}")
        
        # Save checkpoint
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        torch.save(model.state_dict(), OUTPUT_PATH)
        print(f"Saved {OUTPUT_PATH}")

if __name__ == "__main__":
    train()