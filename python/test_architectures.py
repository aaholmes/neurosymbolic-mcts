import torch
import torch.optim as optim
import torch.nn.functional as F
from model_architectures import SmallNet, MinimalViableNet, AlphaZeroNet

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model_shapes(model_class, name):
    print(f"\nTesting {name}...")
    batch_size = 4
    # Random chess board input: [Batch, Channels, Height, Width]
    dummy_input = torch.randn(batch_size, 17, 8, 8)
    
    try:
        model = model_class()
        policy, value = model(dummy_input)
        
        # Check Policy Shape
        assert policy.shape == (batch_size, 4096), f"Policy shape mismatch! Expected ({batch_size}, 4096), got {policy.shape}"
        
        # Check Value Shape
        assert value.shape == (batch_size, 1), f"Value shape mismatch! Expected ({batch_size}, 1), got {value.shape}"
        
        print(f"✅ Shapes Correct. Params: {count_parameters(model):,}")
        return model
    except Exception as e:
        print(f"❌ Failed: {e}")
        return None

def test_overfitting(model_class, name):
    print(f"\nRunning Overfitting Test on {name}...")
    model = model_class()
    model.train() # Explicitly set to train mode
    
    # Use SGD with Momentum for stability on random noise
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    
    # Create a fixed batch of random data
    batch_size = 4 # Smaller batch to make memorization easier
    inputs = torch.randn(batch_size, 17, 8, 8)
    
    # Target Policy: Random probabilities
    target_policy = torch.randn(batch_size, 4096).softmax(dim=1)
    
    # Target Value: Random values between -1 and 1
    target_value = torch.tanh(torch.randn(batch_size, 1))

    initial_loss = 0
    final_loss = 0
    final_val_loss = 0
    
    print("  Iter | Total  | PolLoss | ValLoss")
    print("  -----+--------+---------+--------")
    
    for i in range(201):
        optimizer.zero_grad()
        
        pred_policy, pred_value = model(inputs)
        
        # Loss = MSE(Value) + KL_Div(Policy)
        loss_value = F.mse_loss(pred_value, target_value)
        loss_policy = -torch.sum(target_policy * pred_policy) / batch_size
        
        loss = loss_value + loss_policy
        loss.backward()
        optimizer.step()
        
        if i == 0: initial_loss = loss.item()
        if i == 200: 
            final_loss = loss.item()
            final_val_loss = loss_value.item()
        
        if i % 20 == 0:
            print(f"  {i:4d} | {loss.item():.4f} | {loss_policy.item():.4f}  | {loss_value.item():.4f}")

    # Success criteria: Value loss should drop significantly (indicating gradient flow)
    # Policy loss is hard to minimize on random noise due to high entropy
    if final_val_loss < 0.1:
        print(f"✅ Overfitting Successful (Value Loss dropped to {final_val_loss:.4f})")
    else:
        print(f"⚠️  Overfitting Warning: Value Loss didn't drop enough ({final_val_loss:.4f}). Check learning rate or gradient flow.")

if __name__ == "__main__":
    print("=== Verifying Model Architectures ===")
    
    # 1. Test Shapes & Sizes
    test_model_shapes(SmallNet, "SmallNet")
    test_model_shapes(MinimalViableNet, "MinimalViableNet")
    test_model_shapes(AlphaZeroNet, "AlphaZeroNet")
    
    # 2. Test Gradient Flow (Overfitting)
    # If this works, the layers are connected correctly.
    test_overfitting(SmallNet, "SmallNet")
    test_overfitting(MinimalViableNet, "MinimalViableNet")
    test_overfitting(AlphaZeroNet, "AlphaZeroNet")
