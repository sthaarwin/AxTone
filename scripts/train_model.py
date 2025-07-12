"""
Training script for the guitar tab transcription model.

This script trains the TabTranscriptionModel on the GuitarSet dataset.
"""

import os
import time
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data.dataset import GuitarSetDataset, create_guitarset_dataloaders
from src.core.ai.models.tab_models import TabTranscriptionModel
from src.evaluation.metrics import tab_accuracy, tab_f1_score, tab_timing_error

logger = logging.getLogger(__name__)


def train_model(args):
    """Train the tab transcription model."""
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"tab_model_{timestamp}"
    model_dir = output_dir / "models"
    log_dir = output_dir / "logs"
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir)
    
    # Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    logger.info(f"Loading GuitarSet dataset from {args.data_dir}")
    dataloaders = create_guitarset_dataloaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        n_fft=args.n_fft,
        sequence_length=args.sequence_length
    )
    
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    
    # Create the model
    logger.info("Creating model")
    model = TabTranscriptionModel(
        input_channels=6,  # 6 strings for guitar
        mel_bins=args.mel_bins,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_fret=args.max_fret
    )
    model = model.to(device)
    
    # Define loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    logger.info("Starting training")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_losses = []
        train_string_losses = []
        train_fret_losses = []
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            features = batch['features']['mel_spectrograms'].to(device)
            string_targets = batch['tab_targets']['string_targets'].to(device)
            fret_targets = batch['tab_targets']['fret_targets'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            losses = model.calculate_loss(features, string_targets, fret_targets)
            
            # Backward pass
            total_loss = losses['total_loss']
            total_loss.backward()
            
            # Clip gradients
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            
            # Update weights
            optimizer.step()
            
            # Record losses
            train_losses.append(total_loss.item())
            train_string_losses.append(losses['string_loss'].item())
            train_fret_losses.append(losses['fret_loss'].item())
            
            # Log progress
            if batch_idx % args.log_interval == 0:
                logger.info(
                    f"Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} "
                    f"({100. * batch_idx / len(train_loader):.0f}%)]\t"
                    f"Loss: {total_loss.item():.6f}"
                )
                
                # Log to tensorboard
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('train/total_loss', total_loss.item(), global_step)
                writer.add_scalar('train/string_loss', losses['string_loss'].item(), global_step)
                writer.add_scalar('train/fret_loss', losses['fret_loss'].item(), global_step)
        
        # Calculate average losses
        avg_train_loss = np.mean(train_losses)
        avg_train_string_loss = np.mean(train_string_losses)
        avg_train_fret_loss = np.mean(train_fret_losses)
        
        # Log epoch summary
        train_time = time.time() - start_time
        logger.info(
            f"Epoch {epoch} - "
            f"Train Loss: {avg_train_loss:.6f}, "
            f"String Loss: {avg_train_string_loss:.6f}, "
            f"Fret Loss: {avg_train_fret_loss:.6f}, "
            f"Time: {train_time:.2f}s"
        )
        
        # Validation phase
        model.eval()
        val_losses = []
        val_string_losses = []
        val_fret_losses = []
        
        string_accuracies = []
        fret_accuracies = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                features = batch['features']['mel_spectrograms'].to(device)
                string_targets = batch['tab_targets']['string_targets'].to(device)
                fret_targets = batch['tab_targets']['fret_targets'].to(device)
                
                # Forward pass
                losses = model.calculate_loss(features, string_targets, fret_targets)
                
                # Record losses
                val_losses.append(losses['total_loss'].item())
                val_string_losses.append(losses['string_loss'].item())
                val_fret_losses.append(losses['fret_loss'].item())
                
                # Get predictions for accuracy calculation
                predictions = model.predict(features)
                string_probs = predictions['string_probs']
                fret_probs = predictions['fret_probs']
                
                # Calculate accuracy
                string_preds = (string_probs > 0.5).float()
                fret_preds = torch.argmax(fret_probs, dim=3)
                
                # Compute metrics
                batch_string_acc = tab_accuracy(
                    string_preds.cpu().numpy(),
                    string_targets.permute(0, 2, 1).cpu().numpy(),
                    threshold=0.5
                )
                string_accuracies.append(batch_string_acc)
                
                batch_fret_acc = tab_accuracy(
                    fret_preds.cpu().numpy(),
                    fret_targets.permute(0, 2, 1).cpu().numpy(),
                    is_fret=True
                )
                fret_accuracies.append(batch_fret_acc)
        
        # Calculate average validation metrics
        avg_val_loss = np.mean(val_losses)
        avg_val_string_loss = np.mean(val_string_losses)
        avg_val_fret_loss = np.mean(val_fret_losses)
        avg_string_acc = np.mean(string_accuracies)
        avg_fret_acc = np.mean(fret_accuracies)
        
        # Update learning rate scheduler
        scheduler.step(avg_val_loss)
        
        # Log validation results
        logger.info(
            f"Validation: "
            f"Loss: {avg_val_loss:.6f}, "
            f"String Loss: {avg_val_string_loss:.6f}, "
            f"Fret Loss: {avg_val_fret_loss:.6f}, "
            f"String Acc: {avg_string_acc:.4f}, "
            f"Fret Acc: {avg_fret_acc:.4f}"
        )
        
        # Log to tensorboard
        writer.add_scalar('val/total_loss', avg_val_loss, epoch)
        writer.add_scalar('val/string_loss', avg_val_string_loss, epoch)
        writer.add_scalar('val/fret_loss', avg_val_fret_loss, epoch)
        writer.add_scalar('val/string_accuracy', avg_string_acc, epoch)
        writer.add_scalar('val/fret_accuracy', avg_fret_acc, epoch)
        
        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = model_dir / f"best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'config': config,
            }, best_model_path)
            logger.info(f"New best model saved to {best_model_path}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = model_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'config': config,
            }, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    # Save the final model
    final_model_path = model_dir / "final_model.pt"
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_val_loss,
        'config': config,
    }, final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Close tensorboard writer
    writer.close()
    
    return {
        'output_dir': str(output_dir),
        'best_model_path': str(best_model_path),
        'final_model_path': str(final_model_path),
        'best_val_loss': best_val_loss
    }


def main():
    """Main function to parse arguments and start training."""
    parser = argparse.ArgumentParser(description='Train a guitar tab transcription model')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to GuitarSet dataset')
    parser.add_argument('--output-dir', type=str, default='./models/trained',
                        help='Directory to save models and logs')
    
    # Audio processing arguments
    parser.add_argument('--sample-rate', type=int, default=44100,
                        help='Audio sample rate')
    parser.add_argument('--hop-length', type=int, default=512,
                        help='Hop length for feature extraction')
    parser.add_argument('--n-fft', type=int, default=2048,
                        help='FFT size for feature extraction')
    parser.add_argument('--mel-bins', type=int, default=128,
                        help='Number of mel frequency bins')
    parser.add_argument('--sequence-length', type=int, default=256,
                        help='Sequence length for training')
    
    # Model arguments
    parser.add_argument('--hidden-size', type=int, default=256,
                        help='Hidden size for recurrent layers')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of recurrent layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout probability')
    parser.add_argument('--max-fret', type=int, default=24,
                        help='Maximum fret number to predict')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--clip-grad', type=float, default=1.0,
                        help='Gradient clipping threshold')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Logging interval (batches)')
    parser.add_argument('--save-interval', type=int, default=5,
                        help='Model saving interval (epochs)')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA training')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Start training
    train_model(args)


if __name__ == '__main__':
    main()