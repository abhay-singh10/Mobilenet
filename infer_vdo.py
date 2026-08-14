import os
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Import model factory from model.py
from model import build_model

# ==============================================================================
# CONFIGURATION PATHS & PARAMETERS
# ==============================================================================
INPUT_VIDEO_PATH = "./input_video.mp4"                # Path to input video file
OUTPUT_VIDEO_PATH = "./annotated_output_video.mp4"    # Path to save annotated video
OUTPUT_CSV_PATH = "./window_inference_results.csv"    # Path to save CSV log
CHECKPOINT_PATH = "./checkpoints/best_photosensitive_model.pth" # Model weights

THRESHOLD = 0.50        # Probability threshold for hazard detection
WINDOW_SIZE = 16        # 16-frame window (~0.53 seconds at 30 FPS)
STRIDE = 4              # Overlapping window stride (4 frames = 75% overlap)
TARGET_SIZE = (224, 224)


# ==============================================================================
# FEATURE EXTRACTION & PREPROCESSING HELPERS
# ==============================================================================
def compute_relative_luminance(frame_bgr):
    """Calculates linear relative luminance (Channel 0)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) / 255.0
    mask = rgb <= 0.04045
    rgb_linear = np.where(mask, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    L = 0.2126 * rgb_linear[..., 0] + 0.7152 * rgb_linear[..., 1] + 0.0722 * rgb_linear[..., 2]
    return L.astype(np.float32)


def compute_fft_spatial_band(grayscale_frame):
    """Calculates 2D FFT spatial spectrum in 2.0-6.0 cpd band (Channel 2)."""
    f = np.fft.fft2(grayscale_frame)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1.0)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)

    mask = (dist_from_center >= 10) & (dist_from_center <= 45)
    fft_band = magnitude * mask

    fft_norm = cv2.normalize(fft_band, None, 0, 1, cv2.NORM_MINMAX)
    return fft_norm.astype(np.float32)


def resample_video_to_30fps(video_path):
    """Reads video and resamples frames sequentially to 30 FPS at runtime."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open input video: '{video_path}'")

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if input_fps <= 0:
        input_fps = 30.0

    frames_30fps = []
    dt_target = 1.0 / 30.0
    target_time = 0.0
    src_frame_idx = 0

    print(f"Resampling video from {input_fps:.2f} FPS -> 30.0 FPS at runtime...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        src_time_end = (src_frame_idx + 1) / input_fps

        while target_time < src_time_end - 1e-6:
            frames_30fps.append(frame.copy())
            target_time += dt_target

        src_frame_idx += 1

    cap.release()
    print(f"Total resampled 30 FPS frames loaded: {len(frames_30fps)}")
    return frames_30fps


def format_timestamp(seconds):
    """Formats float seconds to MM:SS.mmm string."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    msecs = int(round((seconds - int(seconds)) * 1000))
    return f"{mins:02d}:{secs:02d}.{msecs:03d}"


# ==============================================================================
# MAIN INFERENCE PIPELINE
# ==============================================================================
def run_video_inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print(f"RUNNING RUNTIME INFERENCE ON DEVICE: {device}")
    print("=" * 80)

    # 1. Validate input video and checkpoint
    if not os.path.exists(INPUT_VIDEO_PATH):
        raise FileNotFoundError(f"Input video not found at '{INPUT_VIDEO_PATH}'!")
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at '{CHECKPOINT_PATH}'!")

    # 2. Resample raw video frames to 30 FPS
    raw_frames = resample_video_to_30fps(INPUT_VIDEO_PATH)
    total_frames = len(raw_frames)

    if total_frames < WINDOW_SIZE:
        raise ValueError(f"Video is too short ({total_frames} frames). Minimum required is {WINDOW_SIZE} frames.")

    # 3. Build 3-channel preprocessed composite maps for all 30 FPS frames
    print("Extracting preprocessed feature channels (L, DeltaL, FFT)...")
    preprocessed_frames = []
    prev_luminance = None

    for frame in raw_frames:
        frame_resized = cv2.resize(frame, TARGET_SIZE)

        # Channel 0: Luminance
        L = compute_relative_luminance(frame_resized)

        # Channel 1: Temporal Difference
        if prev_luminance is None:
            delta_L = np.zeros_like(L)
        else:
            delta_L = np.abs(L - prev_luminance)
        prev_luminance = L

        # Channel 2: Spatial Frequency Map
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        fft_map = compute_fft_spatial_band(gray)

        composite = np.stack([L, delta_L, fft_map], axis=0) # (3, 224, 224)
        preprocessed_frames.append(composite)

    preprocessed_array = np.array(preprocessed_frames, dtype=np.float32) # (Total_Frames, 3, 224, 224)

    # 4. Load trained PyTorch model (num_classes=1 for unified hazard detection)
    print("Loading trained model architecture (num_classes=1)...")
    model = build_model(num_classes=1, num_segments=WINDOW_SIZE, pretrained=False).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 5. Initialize tracking structures for overlapping windows and frame results
    csv_rows = []

    # Per-frame max probability tracking (Max Hazard overrides Clean across overlaps)
    frame_max_hazard = np.zeros(total_frames, dtype=np.float32)

    window_number = 1
    print("Running sliding window inference across video stream...")

    with torch.no_grad():
        for start_idx in range(0, total_frames - WINDOW_SIZE + 1, STRIDE):
            end_idx = start_idx + WINDOW_SIZE

            # Extract 16-frame tensor clip: Shape (1, 16, 3, 224, 224)
            clip_tensor = preprocessed_array[start_idx:end_idx]
            clip_tensor = torch.from_numpy(clip_tensor).unsqueeze(0).to(device)

            # Model inference (Single output logit)
            logits = model(clip_tensor)
            p_hazard = float(torch.sigmoid(logits).cpu().item())

            # Timestamps
            start_ts = format_timestamp(start_idx / 30.0)
            end_ts = format_timestamp(end_idx / 30.0)

            # Determine window result string
            is_hazard = p_hazard >= THRESHOLD
            window_result = "Hazard Detected" if is_hazard else "Clean"

            # Record CSV row with exact hazard percentage
            csv_rows.append({
                'start_ts': start_ts,
                'end_ts': end_ts,
                'window_number': window_number,
                'hazard_probability': f"{p_hazard * 100:.2f}%",
                'Result': window_result
            })

            # Update per-frame maximum probabilities for overlapping resolution
            frame_max_hazard[start_idx:end_idx] = np.maximum(frame_max_hazard[start_idx:end_idx], p_hazard)

            window_number += 1

    # Save CSV Log
    df_csv = pd.DataFrame(csv_rows)
    df_csv.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"CSV inference log saved to: '{OUTPUT_CSV_PATH}'")

    # 6. Render annotated video with top-right overlay
    print("Rendering annotated video with top-right text overlay...")
    height, width, _ = raw_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 30.0, (width, height))

    for frame_idx in range(total_frames):
        frame = raw_frames[frame_idx].copy()

        p_hazard = frame_max_hazard[frame_idx]
        is_hazard = p_hazard >= THRESHOLD

        # Determine label text and color
        if is_hazard:
            display_text = f"HAZARD DETECTED ({p_hazard*100:.1f}%)"
            text_color = (0, 0, 255)      # Red
            bg_color = (0, 0, 100)
        else:
            display_text = f"Clean ({p_hazard*100:.1f}%)"
            text_color = (0, 255, 0)      # Green
            bg_color = (0, 80, 0)

        # Draw top-right text overlay box
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        (text_w, text_h), baseline = cv2.getTextSize(display_text, font, font_scale, thickness)

        # Position at top-right
        margin = 15
        x2 = width - margin
        x1 = x2 - text_w - 20
        y1 = margin
        y2 = y1 + text_h + 20

        # Draw background rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), text_color, 1)

        # Draw text
        text_x = x1 + 10
        text_y = y1 + text_h + 8
        cv2.putText(frame, display_text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

        out_video.write(frame)

    out_video.release()
    print("=" * 80)
    print(f"INFERENCE COMPLETE!")
    print(f"Annotated Video Saved: '{OUTPUT_VIDEO_PATH}'")
    print(f"Results CSV Saved:    '{OUTPUT_CSV_PATH}'")
    print("=" * 80)


if __name__ == "__main__":
    run_video_inference()
