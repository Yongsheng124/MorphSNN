import torch
from torch.utils.data import Dataset
import os
import glob


class UCF101DVS_ProcessedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Points to the preprocessed root directory (e.g., '.../UCF101DVS_Processed')
            transform (callable, optional): Optional transform/data augmentation function
        """
        self.root_dir = root_dir
        self.transform = transform

        # 1. Scan categories
        # Ensure only folders are read
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.files = []

        # 2. Collect all .pt file paths
        # Here only indexing is done, not reading data, so it's fast
        for cls_name in self.classes:
            cls_folder = os.path.join(root_dir, cls_name)
            # Match all .pt files
            pt_files = glob.glob(os.path.join(cls_folder, "*.pt"))
            for f_path in pt_files:
                self.files.append((f_path, self.class_to_idx[cls_name]))

        print(f"Dataset initialization complete: Found {len(self.files)} samples.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f_path, label = self.files[idx]

        # 1. Load Tensor (CPU read)
        # Assume it was saved as uint8 (0/1) to save space
        frames = torch.load(f_path)

        # 2. Convert to float32 (standard network input)
        frames = frames.float()

        # 3. Data augmentation/transform (key step)
        # If you passed a transform, process the single sample here
        # Note: the shape of frames is usually [T, C, H, W]
        if self.transform:
            frames = self.transform(frames)

        return frames, label