#!/usr/bin/env python3
"""
Chess Neural Network Training Pipeline

This module provides a complete pipeline for training chess neural networks
using human game data. It includes PGN parsing, position extraction,
data preprocessing, and model training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import chess
import chess.pgn
import chess.engine
import os
import json
import random
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import argparse
from tqdm import tqdm
import logging
from chess_net import ChessNet, ChessNetInterface

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ChessPosition:
    """Represents a chess position for training"""
    
    def __init__(self, board: chess.Board, result: float, best_move: Optional[chess.Move] = None):
        self.board = board.copy()
        self.result = result  # 1.0 = white wins, 0.5 = draw, 0.0 = black wins
        self.best_move = best_move
        self.fen = board.fen()
        
    def to_tensor(self) -> np.ndarray:
        """Convert board position to neural network input tensor"""
        tensor = np.zeros((17, 8, 8), dtype=np.float32)
        
        # Map piece types to channels
        piece_map = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                rank, file = divmod(square, 8)
                tensor_rank = 7 - rank
                color_offset = 0 if piece.color == chess.WHITE else 6
                channel = color_offset + piece_map[piece.piece_type]
                tensor[channel, tensor_rank, file] = 1.0

        # 12: En Passant
        if self.board.ep_square is not None:
            rank, file = divmod(self.board.ep_square, 8)
            tensor_rank = 7 - rank
            tensor[12, tensor_rank, file] = 1.0
            
        # 13-16: Castling
        if self.board.has_kingside_castling_rights(chess.WHITE):
            tensor[13, :, :] = 1.0
        if self.board.has_queenside_castling_rights(chess.WHITE):
            tensor[14, :, :] = 1.0
        if self.board.has_kingside_castling_rights(chess.BLACK):
            tensor[15, :, :] = 1.0
        if self.board.has_queenside_castling_rights(chess.BLACK):
            tensor[16, :, :] = 1.0
                
        return tensor
    
    def get_move_target(self) -> Optional[int]:
        """Get target move index for policy training"""
        if self.best_move is None:
            return None
        
        # Convert move to from-to encoding (64*64)
        from_square = self.best_move.from_square
        to_square = self.best_move.to_square
        return from_square * 64 + to_square


class ChessDataset(Dataset):
    """PyTorch dataset for chess positions"""
    
    def __init__(self, positions: List[ChessPosition]):
        self.positions = positions
        
    def __len__(self):
        return len(self.positions)
        
    def __getitem__(self, idx):
        pos = self.positions[idx]
        
        # Convert position to tensor
        board_tensor = torch.from_numpy(pos.to_tensor())
        
        # Value target (game result)
        value_target = torch.tensor([pos.result * 2 - 1], dtype=torch.float32)  # Convert to [-1, 1]
        
        # Policy target (one-hot encoded move)
        policy_target = torch.zeros(4096, dtype=torch.float32)
        move_idx = pos.get_move_target()
        if move_idx is not None and move_idx < 4096:
            policy_target[move_idx] = 1.0
            
        return board_tensor, policy_target, value_target


class PGNProcessor:
    """Processes PGN files to extract training positions"""
    
    def __init__(self, min_elo: int = 1800, max_positions_per_game: int = 20):
        self.min_elo = min_elo
        self.max_positions_per_game = max_positions_per_game
        
    def process_pgn_file(self, pgn_path: str, max_games: Optional[int] = None) -> List[ChessPosition]:
        """Extract training positions from a PGN file"""
        positions = []
        games_processed = 0
        
        logger.info(f"Processing PGN file: {pgn_path}")
        
        with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                    
                if max_games and games_processed >= max_games:
                    break
                    
                game_positions = self.extract_positions_from_game(game)
                positions.extend(game_positions)
                games_processed += 1
                
                if games_processed % 100 == 0:
                    logger.info(f"Processed {games_processed} games, extracted {len(positions)} positions")
                    
        logger.info(f"Finished processing {games_processed} games, total positions: {len(positions)}")
        return positions
    
    def extract_positions_from_game(self, game) -> List[ChessPosition]:
        """Extract training positions from a single game"""
        positions = []
        
        # Check if game meets quality criteria
        if not self.is_quality_game(game):
            return positions
            
        # Get game result
        result = self.parse_result(game.headers.get("Result", "*"))
        if result is None:
            return positions
            
        # Walk through the game and extract positions
        board = game.board()
        moves = list(game.mainline_moves())
        
        # Sample positions throughout the game (not every move)
        if len(moves) > self.max_positions_per_game:
            # Sample moves evenly throughout the game
            step = len(moves) // self.max_positions_per_game
            sampled_indices = list(range(0, len(moves), step))[:self.max_positions_per_game]
        else:
            sampled_indices = list(range(len(moves)))
            
        for i, move_idx in enumerate(sampled_indices):
            if move_idx < len(moves):
                # Apply moves up to this position
                temp_board = game.board()
                for j in range(move_idx):
                    temp_board.push(moves[j])
                
                # Only include quiet positions (not in check, not after captures)
                if self.is_quiet_position(temp_board, moves[move_idx] if move_idx < len(moves) else None):
                    best_move = moves[move_idx] if move_idx < len(moves) else None
                    position = ChessPosition(temp_board, result, best_move)
                    positions.append(position)
        
        return positions
    
    def is_quality_game(self, game) -> bool:
        """Check if game meets quality criteria for training"""
        headers = game.headers
        
        # Check player ratings
        try:
            white_elo = int(headers.get("WhiteElo", "0"))
            black_elo = int(headers.get("BlackElo", "0"))
            if white_elo < self.min_elo or black_elo < self.min_elo:
                return False
        except (ValueError, TypeError):
            return False
            
        # Check time control (avoid bullet games)
        time_control = headers.get("TimeControl", "")
        if time_control and "+" in time_control:
            try:
                base_time = int(time_control.split("+")[0])
                if base_time < 300:  # Less than 5 minutes
                    return False
            except:
                pass
        
        # Check that game has reasonable number of moves
        moves = list(game.mainline_moves())
        if len(moves) < 20 or len(moves) > 150:
            return False
            
        return True
    
    def is_quiet_position(self, board: chess.Board, next_move: Optional[chess.Move]) -> bool:
        """Check if position is suitable for training (quiet, not tactical)"""
        # Skip positions where king is in check
        if board.is_check():
            return False
            
        # Skip positions after captures (if we know the next move)
        if next_move and board.is_capture(next_move):
            return False
            
        # Skip positions with very few pieces (endgame tablebases are better)
        piece_count = len(board.piece_map())
        if piece_count < 8:
            return False
            
        return True
    
    def parse_result(self, result_str: str) -> Optional[float]:
        """Parse game result string to numeric value"""
        if result_str == "1-0":
            return 1.0  # White wins
        elif result_str == "0-1":
            return 0.0  # Black wins
        elif result_str == "1/2-1/2":
            return 0.5  # Draw
        else:
            return None  # Unknown result


class ChessTrainer:
    """Handles neural network training"""
    
    def __init__(self, model: ChessNet, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)
        
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        batch_count = 0
        
        for batch_idx, (boards, policy_targets, value_targets) in enumerate(tqdm(dataloader, desc="Training")):
            boards = boards.to(self.device)
            policy_targets = policy_targets.to(self.device)
            value_targets = value_targets.to(self.device)
            
            # Forward pass
            policy_pred, value_pred = self.model(boards)
            
            # Calculate losses
            policy_loss = F.kl_div(policy_pred, policy_targets, reduction='batchmean')
            value_loss = F.mse_loss(value_pred, value_targets)
            total_loss_batch = policy_loss + value_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += total_loss_batch.item()
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
            batch_count += 1
            
        return {
            'total_loss': total_loss / batch_count,
            'policy_loss': policy_loss_sum / batch_count,
            'value_loss': value_loss_sum / batch_count
        }
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        batch_count = 0
        
        with torch.no_grad():
            for boards, policy_targets, value_targets in tqdm(dataloader, desc="Validating"):
                boards = boards.to(self.device)
                policy_targets = policy_targets.to(self.device)
                value_targets = value_targets.to(self.device)
                
                policy_pred, value_pred = self.model(boards)
                
                policy_loss = F.kl_div(policy_pred, policy_targets, reduction='batchmean')
                value_loss = F.mse_loss(value_pred, value_targets)
                total_loss_batch = policy_loss + value_loss
                
                total_loss += total_loss_batch.item()
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                batch_count += 1
                
        return {
            'total_loss': total_loss / batch_count,
            'policy_loss': policy_loss_sum / batch_count,
            'value_loss': value_loss_sum / batch_count
        }
    
    def save_checkpoint(self, path: str, epoch: int, train_loss: float, val_loss: float):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")


def main():
    parser = argparse.ArgumentParser(description="Train chess neural network")
    parser.add_argument("--pgn", type=str, required=True, help="Path to PGN file")
    parser.add_argument("--output", type=str, default="chess_model.pth", help="Output model path")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--max-games", type=int, help="Maximum games to process")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Process PGN file
    logger.info("Processing PGN file...")
    processor = PGNProcessor(min_elo=1800)
    positions = processor.process_pgn_file(args.pgn, max_games=args.max_games)
    
    if len(positions) == 0:
        logger.error("No positions extracted from PGN file")
        return
        
    logger.info(f"Extracted {len(positions)} training positions")
    
    # Split data
    random.shuffle(positions)
    val_size = int(len(positions) * args.val_split)
    train_positions = positions[:-val_size] if val_size > 0 else positions
    val_positions = positions[-val_size:] if val_size > 0 else []
    
    logger.info(f"Training positions: {len(train_positions)}")
    logger.info(f"Validation positions: {len(val_positions)}")
    
    # Create datasets
    train_dataset = ChessDataset(train_positions)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    val_loader = None
    if val_positions:
        val_dataset = ChessDataset(val_positions)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create model and trainer
    model = ChessNet()
    trainer = ChessTrainer(model, device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_metrics = trainer.train_epoch(train_loader)
        logger.info(f"Train - Loss: {train_metrics['total_loss']:.4f}, "
                   f"Policy: {train_metrics['policy_loss']:.4f}, "
                   f"Value: {train_metrics['value_loss']:.4f}")
        
        # Validate
        if val_loader:
            val_metrics = trainer.validate(val_loader)
            logger.info(f"Val - Loss: {val_metrics['total_loss']:.4f}, "
                       f"Policy: {val_metrics['policy_loss']:.4f}, "
                       f"Value: {val_metrics['value_loss']:.4f}")
            
            # Save best model
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                trainer.save_checkpoint(args.output, epoch, 
                                      train_metrics['total_loss'], 
                                      val_metrics['total_loss'])
        else:
            # Save model every few epochs if no validation
            if (epoch + 1) % 5 == 0:
                trainer.save_checkpoint(args.output, epoch, train_metrics['total_loss'], 0.0)
        
        # Update learning rate
        trainer.scheduler.step()
    
    logger.info("Training completed!")
    logger.info(f"Best model saved to: {args.output}")


if __name__ == "__main__":
    main()