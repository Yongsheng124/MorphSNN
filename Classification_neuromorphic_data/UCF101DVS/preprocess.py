import os
import torch
import torch.nn.functional as F
import scipy.io as sio
import glob
from tqdm import tqdm  # Progress bar library, recommend pip install tqdm
import numpy as np

# =================Configuration Area=================
SOURCE_ROOT = "E:\\datasets/UCF101DVS/UCF101DVS"  # Original .mat path
TARGET_ROOT = "E:\\datasets/UCF101DVS/UCF101DVS_Processed"  # Path to save preprocessed data
T = 5
H = 128
W = 128


# =========================================

def events_to_frames(x, y, t, p, T, H, W):
    # (Copy the function from the original class, slightly modify the interface)
    t_start = t.min()
    t_end = t.max()
    total_time = t_end - t_start
    if total_time == 0: return torch.zeros([T, 2, H, W], dtype=torch.float32)
    dt = total_time / T

    # Dynamically get original dimensions
    orig_H = int(y.max()) + 1
    orig_W = int(x.max()) + 1
    orig_H = max(orig_H, 10)
    orig_W = max(orig_W, 10)

    frames = torch.zeros([T, 2, orig_H, orig_W], dtype=torch.float32)

    t_idx = ((t - t_start) / (dt + 1e-6)).astype(int)
    t_idx[t_idx >= T] = T - 1

    x_t = torch.from_numpy(x).long()
    y_t = torch.from_numpy(y).long()
    t_idx_t = torch.from_numpy(t_idx).long()
    p_t = torch.from_numpy(p).long()
    if p_t.min() < 0: p_t[p_t < 0] = 0

    for i in range(T):
        mask = (t_idx_t == i)
        if mask.sum() > 0:
            coords = (p_t[mask], y_t[mask], x_t[mask])
            vals = torch.ones(mask.sum(), dtype=torch.float32)
            frames[i].index_put_(coords, vals, accumulate=True)

    # Resize
    frames = F.interpolate(frames, size=(H, W), mode='bilinear', align_corners=False)
    # Binarize and convert to float
    frames = (frames > 0).float()

    # [Optimization] To save disk space, can save as uint8 (0 or 1), convert to float when reading
    # But for convenience, here directly save as float32 or bool
    return frames.to(torch.uint8)  # Save as integer to save space


def process_all():
    if not os.path.exists(TARGET_ROOT):
        os.makedirs(TARGET_ROOT)

    classes = sorted([d for d in os.listdir(SOURCE_ROOT) if os.path.isdir(os.path.join(SOURCE_ROOT, d))])

    print(f"Starting preprocessing... Target: {TARGET_ROOT}")

    for cls_name in tqdm(classes):
        # 1. Create target class folder
        target_cls_dir = os.path.join(TARGET_ROOT, cls_name)
        if not os.path.exists(target_cls_dir):
            os.makedirs(target_cls_dir)

        source_cls_dir = os.path.join(SOURCE_ROOT, cls_name)
        mat_files = glob.glob(os.path.join(source_cls_dir, "*.mat"))

        for f_path in mat_files:
            try:
                # Read
                data = sio.loadmat(f_path)
                # Adapt to your data structure
                if 'pol' in data:  # Adjust based on your previous data structure
                    x = data['x'].flatten()
                    y = data['y'].flatten()
                    t = data['ts'].flatten()
                    p = data['pol'].flatten()
                else:
                    # Fallback
                    x = data['x'].flatten()
                    y = data['y'].flatten()
                    t = data['t'].flatten()
                    p = data['p'].flatten()

                # Process
                frames = events_to_frames(x, y, t, p, T, H, W)

                # Save: Keep filename consistent, change extension to .pt
                file_name = os.path.basename(f_path).replace('.mat', '.pt')
                save_path = os.path.join(target_cls_dir, file_name)

                torch.save(frames, save_path)

            except Exception as e:
                print(f"Skipping {f_path}: {e}")


if __name__ == "__main__":
    process_all()