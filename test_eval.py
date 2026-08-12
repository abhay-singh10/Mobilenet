import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score, classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Import your model architecture factory from model.py
from model import build_model

# ==============================================================================
# CONFIGURATION PATHS & PARAMETERS
# ==============================================================================
TEST_MANIFEST_CSV = "./test_manifest.csv"             # Path to your test dataset manifest CSV
CHECKPOINT_PATH = "./checkpoints/best_photosensitive_model.pth" # Trained model weights
OUTPUT_RESULTS_CSV = "./test_predictions_results.csv"  # File to save detailed per-clip predictions

BATCH_SIZE = 32                                       # Batch size for testing
DECISION_THRESHOLD = 0.50                             # Probability threshold for positive hazard class
NUM_WORKERS = 4                                       # DataLoader CPU worker threads


# ==============================================================================
# PYTORCH TEST DATASET CLASS
# ==============================================================================
class PhotosensitiveTestDataset(Dataset):
    """
    PyTorch Dataset loading preprocessed testing tensors (16, 3, 224, 224)
    and ground-truth binary targets [flicker_label, illusion_label].
    """
    def __init__(self, df_manifest):
        self.df = df_manifest.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load preprocessed tensor chunk: Shape (16, 3, 224, 224)
        tensor_path = row['tensor_path']
        tensor_array = np.load(tensor_path).astype(np.float32)
        tensor = torch.from_numpy(tensor_array)
        
        # Multi-label ground truth vector: [flicker_label, illusion_label]
        labels = np.array([row['flicker_label'], row['illusion_label']], dtype=np.float32)
        target = torch.from_numpy(labels)
        
        return tensor, target, tensor_path


# ==============================================================================
# MAIN TEST & EVALUATION EXECUTION
# ==============================================================================
def run_testing():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print(f"RUNNING MODEL TEST EVALUATION ON DEVICE: {device}")
    print("=" * 80)

    # 1. Verify file paths exist
    if not os.path.exists(TEST_MANIFEST_CSV):
        raise FileNotFoundError(f"Test manifest CSV not found at '{TEST_MANIFEST_CSV}'!")
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Trained model checkpoint not found at '{CHECKPOINT_PATH}'!")

    # 2. Load test manifest
    test_df = pd.read_csv(TEST_MANIFEST_CSV)
    print(f"Loaded testing manifest with {len(test_df)} samples.")

    test_dataset = PhotosensitiveTestDataset(test_df)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True if device.type == 'cuda' else False
    )

    # 3. Instantiate model architecture and load trained weights
    print("\nInitializing model architecture and loading trained weights...")
    model = build_model(num_classes=2, num_segments=16, pretrained=False).to(device)
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded checkpoint trained up to Epoch {checkpoint.get('epoch', 'N/A')}")

    # 4. Run Inference Loop
    all_targets = []
    all_probs = []
    all_paths = []

    start_time = time.time()
    print("\nExecuting inference loop across test samples...")

    with torch.no_grad():
        for inputs, targets, paths in test_loader:
            inputs = inputs.to(device)  # Shape: (B, 16, 3, 224, 224)
            
            # Forward pass
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy()  # Convert logits to probabilities [0.0, 1.0]

            all_probs.append(probs)
            all_targets.append(targets.numpy())
            all_paths.extend(paths)

    total_time = time.time() - start_time
    time_per_clip = (total_time / len(test_df)) * 1000  # in ms
    time_per_frame = time_per_clip / 16                # in ms

    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)
    all_preds = (all_probs >= DECISION_THRESHOLD).astype(int)

    print(f"Inference Complete in {total_time:.2f} seconds!")
    print(f"Avg Processing Latency per 16-frame Clip: {time_per_clip:.2f} ms")
    print(f"Avg Processing Latency per Single Frame: {time_per_frame:.2f} ms")

    # 5. Calculate Classification Metrics
    exact_match_acc = accuracy_score(all_targets, all_preds)
    
    # Precision, Recall, F1 per class
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_preds, average=None, zero_division=0
    )

    # ROC-AUC per class
    try:
        flicker_auc = roc_auc_score(all_targets[:, 0], all_probs[:, 0])
    except ValueError:
        flicker_auc = 0.0

    try:
        illusion_auc = roc_auc_score(all_targets[:, 1], all_probs[:, 1])
    except ValueError:
        illusion_auc = 0.0

    # 6. Print Professional Evaluation Summary
    print("\n" + "=" * 80)
    print("MODEL EVALUATION RESULTS METRICS SUMMARY")
    print("=" * 80)
    print(f"Decision Probability Threshold: {DECISION_THRESHOLD}")
    print(f"Subset Accuracy (Exact Match Across Both Targets): {exact_match_acc * 100:.2f}%\n")

    metrics_df = pd.DataFrame({
        'Hazard Class': ['Flicker / Strobe Hazard', 'Optical Illusion Hazard'],
        'Support (Samples)': [int(support[0]), int(support[1])],
        'Precision': [f"{precision[0]:.4f}", f"{precision[1]:.4f}"],
        'Recall': [f"{recall[0]:.4f}", f"{recall[1]:.4f}"],
        'F1-Score': [f"{f1[0]:.4f}", f"{f1[1]:.4f}"],
        'ROC-AUC': [f"{flicker_auc:.4f}", f"{illusion_auc:.4f}"]
    })
    print(metrics_df.to_string(index=False))

    print("\n--- Detailed Scikit-Learn Classification Report ---")
    print("Target 0: Flicker / Strobe Hazard")
    print(classification_report(all_targets[:, 0], all_preds[:, 0], target_names=['Safe', 'Flicker Hazard'], zero_division=0))
    
    print("Target 1: Optical Illusion / Hypnotic Hazard")
    print(classification_report(all_targets[:, 1], all_preds[:, 1], target_names=['Safe', 'Illusion Hazard'], zero_division=0))

    # 7. Export Detailed Predictions to CSV
    results_df = pd.DataFrame({
        'tensor_path': all_paths,
        'true_flicker': all_targets[:, 0].astype(int),
        'true_illusion': all_targets[:, 1].astype(int),
        'pred_flicker_prob': np.round(all_probs[:, 0], 4),
        'pred_illusion_prob': np.round(all_probs[:, 1], 4),
        'pred_flicker_label': all_preds[:, 0],
        'pred_illusion_label': all_preds[:, 1],
        'flicker_correct': (all_targets[:, 0] == all_preds[:, 0]).astype(int),
        'illusion_correct': (all_targets[:, 1] == all_preds[:, 1]).astype(int)
    })

    results_df.to_csv(OUTPUT_RESULTS_CSV, index=False)
    print("\n" + "=" * 80)
    print(f"Detailed prediction log saved to: '{OUTPUT_RESULTS_CSV}'")
    print("=" * 80)


if __name__ == "__main__":
    run_testing()
