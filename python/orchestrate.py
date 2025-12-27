import os
import subprocess
import torch
import shutil
from model import LogosNet

# Configuration
GAMES_PER_GEN = 10 # 500 in prod, 10 for testing
SIMULATIONS = 100 # 800 in prod
TRAIN_EPOCHS = 2
BUFFER_SIZE = 20

def export_model_for_rust(model, output_path):
    """
    Export PyTorch model to TorchScript for Rust integration.
    Must handle the dual input signature: (board, material_scalar)
    """
    model.eval()
    # Dummy inputs
    example_board = torch.randn(1, 12, 8, 8)
    example_material = torch.randn(1, 1)
    
    traced_script_module = torch.jit.trace(model, (example_board, example_material))
    traced_script_module.save(output_path)
    print(f"Exported TorchScript model to {output_path}")

def initialize_generation_0():
    weights_dir = "weights"
    os.makedirs(weights_dir, exist_ok=True)
    gen_0_path = os.path.join(weights_dir, "gen_0.pt")
    gen_0_pth = os.path.join(weights_dir, "gen_0.pth") # State dict for Python

    if not os.path.exists(gen_0_path):
        print("Initializing Generation 0...")
        model = LogosNet()
        # Export for Rust
        export_model_for_rust(model, gen_0_path)
        # Export for Python
        torch.save(model.state_dict(), gen_0_pth)

def orchestrate():
    generation = 1
    
    # Ensure directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("weights", exist_ok=True)

    initialize_generation_0()

    while True:
        print(f"\n=== Starting Generation {generation} ===")
        
        # Paths
        prev_model_pt = f"weights/gen_{generation-1}.pt"
        prev_model_pth = f"weights/gen_{generation-1}.pth"
        current_data_dir = f"data/gen_{generation}"
        next_model_pt = f"weights/gen_{generation}.pt"
        next_model_pth = f"weights/gen_{generation}.pth"

        # 1. Self Play
        print(f"🎮 Generating {GAMES_PER_GEN} games...")
        # cargo run --bin self_play -- <games> <sims> <output_dir> <model_path>
        cmd = [
            "cargo", "run", "--release", "--bin", "self_play", "--",
            str(GAMES_PER_GEN), str(SIMULATIONS), current_data_dir, prev_model_pt
        ]
        subprocess.check_call(cmd)

        # 2. Training
        print(f"🧠 Training on last {BUFFER_SIZE} generations...")
        
        # Aggregate data folders (Rolling Window)
        # For simplicity in this v1, pass the current gen folder. 
        # python/train.py logic needs to change to accept multiple folders if we strictly follow the guide.
        # OR we just copy files into a 'training_buffer' dir.
        buffer_dir = "data/training_buffer"
        if os.path.exists(buffer_dir):
            shutil.rmtree(buffer_dir)
        os.makedirs(buffer_dir)
        
        start_gen = max(1, generation - BUFFER_SIZE + 1)
        for g in range(start_gen, generation + 1):
            src = f"data/gen_{g}"
            if os.path.exists(src):
                # Symlink or copy files
                for f in os.listdir(src):
                    if f.endswith(".bin"):
                        shutil.copy(os.path.join(src, f), os.path.join(buffer_dir, f"{g}_{f}"))

        # Call train.py
        # python train.py <data_dir> <output_model_path> <resume_path>
        cmd = [
            "python3", "python/train.py",
            buffer_dir, next_model_pth, prev_model_pth
        ]
        subprocess.check_call(cmd)

        # 3. Export to TorchScript for Rust
        print("📦 Exporting model for engine...")
        model = LogosNet()
        model.load_state_dict(torch.load(next_model_pth))
        export_model_for_rust(model, next_model_pt)
        
        # 4. Update Symlink
        if os.path.exists("weights/latest.pt"):
            os.remove("weights/latest.pt")
        os.symlink(os.path.abspath(next_model_pt), "weights/latest.pt")
        
        print(f"✅ Generation {generation} Complete!")
        generation += 1

if __name__ == "__main__":
    orchestrate()
