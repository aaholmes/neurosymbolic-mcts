import torch
from model_architectures import MinimalViableNet
import os

def export_model():
    print("🧠 Exporting MinimalViableNet to TorchScript...")
    
    # 1. Instantiate the model
    model = MinimalViableNet()
    model.eval() # Set to inference mode
    
    # 2. Create dummy input for tracing/verification
    # Shape: [Batch, Channels, 8, 8]
    dummy_input = torch.randn(1, 17, 8, 8)
    
    # 3. Trace the model
    # Tracing runs the model once and records the operations.
    # It is generally robust for standard CNNs.
    try:
        traced_model = torch.jit.trace(model, dummy_input)
        
        # Create output directory
        os.makedirs("models", exist_ok=True)
        output_path = "models/model.pt"
        
        # 4. Save
        traced_model.save(output_path)
        print(f"✅ Successfully exported model to: {output_path}")
        
        # Verification
        print("   Verifying saved model...")
        loaded = torch.jit.load(output_path)
        p, v = loaded(dummy_input)
        print(f"   Inference Check - Policy Shape: {p.shape}, Value Shape: {v.shape}")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")

if __name__ == "__main__":
    export_model()
