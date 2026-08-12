import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# ==============================================================================
# CONFIGURATION PATHS & PARAMETERS
# ==============================================================================
VIDEOS_DIR = "./videos_30fps"          # Directory containing 30 FPS videos
CSV_ANNOTATIONS = "./annotations.csv"   # Path to your timestamp CSV file
OUTPUT_TENSORS_DIR = "./data_tensors"  # Directory where .npy files will be saved
OUTPUT_MANIFEST = "./dataset_manifest.csv" # Final CSV for PyTorch DataLoader

CHUNK_SIZE = 16   # 16 frames per sample (~0.53 sec at 30 FPS)
STRIDE = 8        # 8 frames stride (50% overlap for data augmentation)
TARGET_SIZE = (224, 224)


def compute_relative_luminance(frame_bgr):
    """Computes normalized linear relative luminance (Channel 0)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) / 255.0
    mask = rgb <= 0.04045
    rgb_linear = np.where(mask, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    L = 0.2126 * rgb_linear[..., 0] + 0.7152 * rgb_linear[..., 1] + 0.0722 * rgb_linear[..., 2]
    return L.astype(np.float32)


def compute_fft_spatial_band(grayscale_frame):
    """Computes 2D FFT spatial frequency spectrum in 2.0-6.0 cpd band (Channel 2)."""
    f = np.fft.fft2(grayscale_frame)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1.0)
    
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    # Mask 2.0 - 6.0 cpd band at 224x224
    mask = (dist_from_center >= 10) & (dist_from_center <= 45)
    fft_band = magnitude * mask
    
    fft_norm = cv2.normalize(fft_band, None, 0, 1, cv2.NORM_MINMAX)
    return fft_norm.astype(np.float32)


def process_video_frames(video_path):
    """Extracts preprocessed 3-channel feature maps for all frames in a video."""
    cap = cv2.VideoCapture(str(video_path))
    processed_frames = []
    prev_luminance = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to 224x224
        frame_resized = cv2.resize(frame, TARGET_SIZE)

        # Channel 0: Luminance (L)
        L = compute_relative_luminance(frame_resized)

        # Channel 1: Temporal Difference (|L_t - L_{t-1}|)
        if prev_luminance is None:
            delta_L = np.zeros_like(L)
        else:
            delta_L = np.abs(L - prev_luminance)
        prev_luminance = L

        # Channel 2: Spatial Frequency Map (FFT)
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        fft_map = compute_fft_spatial_band(gray)

        # Stack into 3-channel composite frame: Shape (3, 224, 224)
        composite_frame = np.stack([L, delta_L, fft_map], axis=0)
        processed_frames.append(composite_frame)

    cap.release()
    if len(processed_frames) == 0:
        return None
    
    # Shape: (Total_Frames, 3, 224, 224)
    return np.array(processed_frames, dtype=np.float32)


def generate_frame_level_labels(total_frames, video_annotations):
    """Creates boolean hazard lookup arrays for every frame index in the video."""
    flicker_mask = np.zeros(total_frames, dtype=int)
    illusion_mask = np.zeros(total_frames, dtype=int)

    for _, row in video_annotations.iterrows():
        start_idx = int(round(row['start_sec'] * 30.0))
        end_idx = int(round(row['end_sec'] * 30.0))
        
        # Clamp bounds
        start_idx = max(0, start_idx)
        end_idx = min(total_frames, end_idx)

        if row.get('is_flicker', 0) == 1:
            flicker_mask[start_idx:end_idx] = 1
        if row.get('is_illusion', 0) == 1:
            illusion_mask[start_idx:end_idx] = 1

    return flicker_mask, illusion_mask


def build_dataset():
    output_tensor_dir = Path(OUTPUT_TENSORS_DIR)
    output_tensor_dir.mkdir(parents=True, exist_ok=True)

    # Load annotation CSV
    df_annotations = pd.read_csv(CSV_ANNOTATIONS)
    
    manifest_rows = []
    video_files = list(Path(VIDEOS_DIR).glob("*.*"))

    print(f"Found {len(video_files)} video file(s) to process.")

    for vid_idx, video_path in enumerate(video_files, start=1):
        video_name = video_path.name
        print(f"[{vid_idx}/{len(video_files)}] Processing video: {video_name}...")

        # 1. Process video into preprocessed composite tensor array
        video_features = process_video_frames(video_path)
        if video_features is None:
            print(f" -> Warning: Could not read video '{video_name}', skipping.")
            continue

        total_frames = video_features.shape[0]
        if total_frames < CHUNK_SIZE:
            print(f" -> Warning: Video '{video_name}' is shorter than {CHUNK_SIZE} frames ({total_frames} frames), skipping.")
            continue

        # 2. Extract timestamp labels for this video
        vid_annotations = df_annotations[df_annotations['video_name'] == video_name]
        flicker_mask, illusion_mask = generate_frame_level_labels(total_frames, vid_annotations)

        # 3. Slide 16-frame window across video
        chunk_count = 0
        for start_f in range(0, total_frames - CHUNK_SIZE + 1, STRIDE):
            end_f = start_f + CHUNK_SIZE
            
            # Extract 16-frame tensor chunk: Shape (16, 3, 224, 224)
            chunk_tensor = video_features[start_f:end_f]

            # Majority voting for chunk label (>= 8 frames inside hazard range)
            flicker_active_frames = np.sum(flicker_mask[start_f:end_f])
            illusion_active_frames = np.sum(illusion_mask[start_f:end_f])

            chunk_flicker_label = 1 if flicker_active_frames >= 8 else 0
            chunk_illusion_label = 1 if illusion_active_frames >= 8 else 0

            # Save tensor chunk as .npy file
            save_name = f"{video_path.stem}_chunk_{chunk_count:04d}.npy"
            save_path = output_tensor_dir / save_name
            np.save(save_path, chunk_tensor)

            # Record entry in manifest
            manifest_rows.append({
                'tensor_path': str(save_path.resolve()),
                'flicker_label': chunk_flicker_label,
                'illusion_label': chunk_illusion_label
            })
            chunk_count += 1

        print(f" -> Saved {chunk_count} chunk tensors for '{video_name}'")

    # 4. Save master dataset manifest
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(OUTPUT_MANIFEST, index=False)
    print("\n" + "=" * 50)
    print(f"Dataset preparation complete! Total chunks created: {len(manifest_df)}")
    print(f"Master manifest saved to: '{OUTPUT_MANIFEST}'")


if __name__ == "__main__":
    build_dataset()
