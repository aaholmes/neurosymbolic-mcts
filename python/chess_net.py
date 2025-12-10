#!/usr/bin/env python3
"""
Chess Neural Network Policy for Kingfisher Engine

This module implements a PyTorch neural network for chess position evaluation
and move policy prediction, designed to integrate with the Rust chess engine.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional
import json
import os

# Import architectures from the consolidated module
from model_architectures import MinimalViableNet as ChessNet, ResidualBlock

class ChessNetInterface:
    """
    Interface for loading and using the chess neural network
    Designed for integration with Rust engine via Python API
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = ChessNet().to(self.device)
        self.model.eval()
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            print(f"✅ Loaded chess model from {model_path}")
        else:
            print("⚠️  Using randomly initialized model (no trained weights loaded)")
    
    def load_model(self, path: str):
        """Load trained model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
    
    def save_model(self, path: str, metadata: Optional[dict] = None):
        """Save model weights and metadata"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'metadata': metadata or {}
        }
        torch.save(checkpoint, path)
        print(f"✅ Saved model to {path}")
    
    def board_to_tensor(self, board_array: np.ndarray) -> torch.Tensor:
        """
        Convert board representation to neural network input tensor
        
        Args:
            board_array: numpy array of shape (12, 8, 8) representing piece positions
            
        Returns:
            torch tensor ready for model input
        """
        tensor = torch.from_numpy(board_array).float()
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        return tensor.to(self.device)
    
    def predict(self, board_array: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Predict move policy and position value
        
        Args:
            board_array: Board representation as numpy array (12, 8, 8)
            
        Returns:
            policy: Move probabilities as numpy array (4096,)
            value: Position evaluation as float [-1, 1]
        """
        with torch.no_grad():
            tensor = self.board_to_tensor(board_array)
            policy_logits, value = self.model(tensor)
            
            # Convert to numpy
            policy = torch.exp(policy_logits).cpu().numpy()[0]  # Convert log_softmax to probabilities
            value = value.cpu().numpy()[0, 0]
            
            return policy, value
    
    def get_top_moves(self, board_array: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Get top-k moves with their probabilities
        
        Args:
            board_array: Board representation
            top_k: Number of top moves to return
            
        Returns:
            List of (move_index, probability) tuples sorted by probability
        """
        policy, _ = self.predict(board_array)
        
        # Get top-k indices and their probabilities
        top_indices = np.argsort(policy)[-top_k:][::-1]
        top_moves = [(int(idx), float(policy[idx])) for idx in top_indices]
        
        return top_moves


def create_training_data_sample():
    """Create sample training data for testing"""
    # Create a few sample positions (normally this would come from game data)
    batch_size = 4
    
    # Random board positions (in practice, these would be real chess positions)
    boards = torch.randn(batch_size, 12, 8, 8)
    
    # Random target policies (in practice, these would be human/engine moves)
    target_policies = torch.randn(batch_size, 4096)
    target_policies = F.softmax(target_policies, dim=1)
    
    # Random target values (in practice, these would be game outcomes)
    target_values = torch.randn(batch_size, 1)
    
    return boards, target_policies, target_values


def train_step_example():
    """Example training step (for demonstration)"""
    model = ChessNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Sample data
    boards, target_policies, target_values = create_training_data_sample()
    
    # Forward pass
    pred_policies, pred_values = model(boards)
    
    # Loss calculation
    policy_loss = F.kl_div(pred_policies, target_policies, reduction='batchmean')
    value_loss = F.mse_loss(pred_values, target_values)
    total_loss = policy_loss + value_loss
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    print(f"Policy Loss: {policy_loss.item():.4f}")
    print(f"Value Loss: {value_loss.item():.4f}")
    print(f"Total Loss: {total_loss.item():.4f}")


if __name__ == "__main__":
    print("🧠 Chess Neural Network Module")
    print("==============================")
    
    # Test model creation
    interface = ChessNetInterface()
    print(f"Model parameters: {sum(p.numel() for p in interface.model.parameters()):,}")
    
    # Test prediction with random board
    test_board = np.random.rand(12, 8, 8).astype(np.float32)
    policy, value = interface.predict(test_board)
    
    print(f"Policy shape: {policy.shape}")
    print(f"Value: {value:.4f}")
    print(f"Top 5 moves: {interface.get_top_moves(test_board, 5)}")
    
    # Demo training step
    print("\n🔧 Training Demo:")
    train_step_example()
    
    print("\n✅ Neural network infrastructure ready!")