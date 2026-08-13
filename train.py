import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import your updated model architecture builder from model.py
from model import build_model

# ==============================================================================
# CONFIGURATION & HYPERPARAMETERS
# ==============================================================================
MANIFEST_CSV = "./dataset_manifest.csv"  # Path to manifest CSV with 'is_hazard' column
SAVE_DIR = "./checkpoints"              # Directory to save trained model weights
BATCH_SIZE = 32                         # Batch size (adjust based on GPU VRAM)
NUM_EPOCHS = 25                         # Total training epochs
LEARNING_RATE = 1e-3                    # Initial learning rate
WEIGHT_DECAY = 1e-4                     # AdamW weight decay
VAL_SIZE = 0.2                          # 20% validation split
RANDOM_SEED = 42                        # Seed for reproducible train/val splits
NUM_WORKERS = 4                         # DataLoader CPU worker threads
THRESHOLD = 0.50                        # Decision threshold for hazard detection


# ==============================================================================
# PYTORCH DATASET CLASS
# ==============================================================================
class PhotosensitiveDataset(Dataset):
    """
    PyTorch Dataset that loads pre-saved (16, 3, 224, 224) numpy tensors 
    and single binary target [is_hazard] (0 = Safe, 1 = Hazard).
    """
    def __init__(self, df_manifest):
        self.df = df_manifest.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load preprocessed tensor chunk: Shape (16, 3, 224, 224)
        tensor_array = np.load(row['tensor_path']).astype(np.float32)
        tensor = torch.from_numpy(tensor_array)

        # Single binary target: [is_hazard]
        target = torch.tensor([row['is_hazard']], dtype=torch.float32)

        return tensor, target


# ==============================================================================
# METRIC EVALUATION HELPER
# ==============================================================================
def calculate_metrics(y_true, y_pred_probs, threshold=0.50):
    """
    Calculates binary classification accuracy, precision, recall, and F1-score.
    """
    y_pred = (y_pred_probs >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# ==============================================================================
# TRAIN & VALIDATION LOOPS
# ==============================================================================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    for inputs, targets in dataloader:
        inputs = inputs.to(device)   # Shape: (B, 16, 3, 224, 224)
        targets = targets.to(device) # Shape: (B, 1)

        optimizer.zero_grad()

        # Forward pass
        logits = model(inputs)       # Shape: (B, 1)
        loss = criterion(logits, targets)

        # Backward pass & Optimization
        loss.backward()
        optimizer.step()

        # Track statistics
        running_loss += loss.item() * inputs.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()

        all_probs.append(probs)
        all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)

    metrics = calculate_metrics(all_targets, all_probs, threshold=THRESHOLD)
    metrics['loss'] = epoch_loss
    return metrics


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    for inputs, targets in dataloader:
        inputs = inputs.to(device)   # Shape: (B, 16, 3, 224, 224)
        targets = targets.to(device) # Shape: (B, 1)

        logits = model(inputs)       # Shape: (B, 1)
        loss = criterion(logits, targets)

        running_loss += loss.item() * inputs.size(0)
        probs = torch.sigmoid(logits).cpu().numpy()

        all_probs.append(probs)
        all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)

    metrics = calculate_metrics(all_targets, all_probs, threshold=THRESHOLD)
    metrics['loss'] = epoch_loss
    return metrics


# ==============================================================================
# MAIN TRAINING EXECUTION
# ==============================================================================
def main():
    # 1. Setup execution device and save paths
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing training on Device: {device}")

    save_path = Path(SAVE_DIR)
    save_path.mkdir(parents=True, exist_ok=True)

    # 2. Load dataset manifest & perform train/val split
    if not os.path.exists(MANIFEST_CSV):
        raise FileNotFoundError(f"Manifest CSV not found at '{MANIFEST_CSV}'. Run prepare_dataset.py first!")

    manifest_df = pd.read_csv(MANIFEST_CSV)
    print(f"Loaded manifest with {len(manifest_df)} total samples.")

    # Stratified split ensures balanced ratio of Safe (0) vs Hazard (1) in both sets
    train_df, val_df = train_test_split(
        manifest_df, 
        test_size=VAL_SIZE, 
        random_state=RANDOM_SEED, 
        shuffle=True,
        stratify=manifest_df['is_hazard'] if 'is_hazard' in manifest_df else None
    )
    print(f"Dataset split: {len(train_df)} Training samples | {len(val_df)} Validation samples.")

    # 3. Create PyTorch DataLoaders
    train_dataset = PhotosensitiveDataset(train_df)
    val_dataset = PhotosensitiveDataset(val_df)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=NUM_WORKERS, pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=NUM_WORKERS, pin_memory=True if device.type == 'cuda' else False
    )

    # 4. Instantiate Model (num_classes = 1 for Unified Hazard Detection)
    print("Building MobileNetV3 + Causal TSM Model (num_classes = 1)...")
    model = build_model(num_classes=1, num_segments=16, pretrained=True).to(device)

    # Binary Cross-Entropy with Logits Loss
    criterion = nn.BCEWithLogitsLoss()

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # 5. Training Loop
    best_val_f1 = 0.0
    print("\n" + "=" * 80)
    print("STARTING MODEL TRAINING (UNIFIED HAZARD CLASSIFICATION)")
    print("=" * 80)

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)

        scheduler.step()
        elapsed = time.time() - start_time

        # Print Epoch Summary
        print(f"\nEpoch [{epoch:02d}/{NUM_EPOCHS:02d}] ({elapsed:.1f}s) | LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"  Train -> Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']*100:.2f}% | F1: {train_metrics['f1']:.4f}")
        print(f"  Val   -> Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']*100:.2f}% | F1: {val_metrics['f1']:.4f} | Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f}")

        # Checkpoint Saving: Save best model based on validation F1-score
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_checkpoint_path = save_path / "best_photosensitive_model.pth"

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'hyperparameters': {
                    'batch_size': BATCH_SIZE,
                    'lr': LEARNING_RATE,
                    'num_segments': 16,
                    'num_classes': 1
                }
            }, best_checkpoint_path)

            print(f"  >>> Best model saved! (New Best Val F1: {best_val_f1:.4f})")

    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE! Best Validation F1-Score: {best_val_f1:.4f}")
    print(f"Model saved to: {save_path / 'best_photosensitive_model.pth'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
