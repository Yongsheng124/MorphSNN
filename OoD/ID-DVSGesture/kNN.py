import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.clock_driven import functional
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, fcluster
from spikingjelly.datasets.n_caltech101 import NCaltech101
# Import your model definition
from smodel import Model
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


# === Define Noise Transform ===

class TemporalShuffle(object):
    """Temporal complete shuffle: [T, C, H, W] -> randomly ordered T"""

    def __call__(self, tensor):
        T = tensor.shape[0]
        perm = torch.randperm(T)
        return tensor[perm]


class AddGaussianNoise(object):
    """Inject strong noise: destroy sparsity"""

    def __init__(self, mean=0., std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        # Add noise and ReLU to ensure non-negative
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.relu(tensor + noise)


class SpatialJigsaw(object):
    """Spatial jigsaw shuffle: cut the image and reassemble it"""

    def __init__(self, grid_size=2):
        self.grid_size = grid_size  # 2x2 = 4 pieces

    def __call__(self, tensor):
        # tensor: [T, C, H, W]
        T, C, H, W = tensor.shape
        h_grid = H // self.grid_size
        w_grid = W // self.grid_size

        # 1. Cut into patches
        patches = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Each patch is [T, C, h_grid, w_grid]
                patch = tensor[..., i * h_grid:(i + 1) * h_grid, j * w_grid:(j + 1) * w_grid]
                patches.append(patch)

        # 2. Shuffle
        import random
        random.shuffle(patches)

        # 3. Reassemble (simple processing: concatenate along W or H dimension)
        # Here to maintain [H, W] shape unchanged, we reassemble as grid
        rows = []
        for i in range(self.grid_size):
            # Concatenate a row
            row_patches = patches[i * self.grid_size: (i + 1) * self.grid_size]
            rows.append(torch.cat(row_patches, dim=-1))  # Concatenate along W

        # Concatenate rows
        return torch.cat(rows, dim=-2)  # Concatenate along H


class TimeFlip(object):
    def __call__(self, tensor):
        # tensor: [T, C, H, W]
        # Flip along time dimension (dim=0)
        return torch.flip(tensor, [0])


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.subset[index]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.subset)


# ==========================================
# 0. Dataset utilities (keep your logic)
# ==========================================
class SubsetWithTransform(Dataset):
    def __init__(self, dataset: Dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.dataset[self.indices[i]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y


def split_to_train_test_set(train_ratio, origin_dataset, num_classes, random_split=False, transform_train=None,
                            transform_test=None):
    label_idx = [[] for _ in range(num_classes)]
    if hasattr(origin_dataset, 'targets'):
        targets = origin_dataset.targets
        for i, y in enumerate(targets):
            label_idx[y].append(i)
    else:
        print("Indexing dataset (slow)...")
        for i in tqdm(range(len(origin_dataset))):
            _, y = origin_dataset[i]
            if isinstance(y, (np.ndarray, torch.Tensor)):
                y = y.item()
            label_idx[y].append(i)

    if random_split:
        for i in range(num_classes):
            np.random.shuffle(label_idx[i])

    train_idx, test_idx = [], []
    for i in range(num_classes):
        pos = math.ceil(len(label_idx[i]) * train_ratio)
        train_idx.extend(label_idx[i][:pos])
        test_idx.extend(label_idx[i][pos:])

    train_set = SubsetWithTransform(origin_dataset, train_idx, transform=transform_train)
    test_set = SubsetWithTransform(origin_dataset, test_idx, transform=transform_test)
    return train_set, test_set


def compute_knn_scores(train_features, test_features, k=50):
    """
    Calculate KNN distance scores.
    Score = distance to k-th nearest neighbor (Euclidean Distance).
    """
    print(f"Fitting KNN (k={k}) on {len(train_features)} training samples...")

    # 1. Build KNN index on training features
    # n_jobs=-1 uses all CPU cores for acceleration
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric='euclidean', n_jobs=-1)
    nbrs.fit(train_features)

    print(f"Querying KNN for {len(test_features)} test samples...")
    # 2. Query k nearest neighbors for test samples
    # distances shape: [num_samples, k], containing distances to k nearest points, sorted from small to large
    distances, _ = nbrs.kneighbors(test_features)

    # 3. Take the distance to the k-th nearest neighbor as anomaly score
    # Theoretically could also take average of k distances (np.mean(distances, axis=1)), but in OoD commonly use k-th
    kth_distance = distances[:, -1]

    return kth_distance


def run_knn_analysis(train_features, id_test_features, ood_features, k=50):
    print(f"\n>>> Starting KNN evaluation (k={k})...")

    # 1. Data preprocessing: ensure 2D (N, D)
    # Check dimensions, if 3D [N, T, D], compress time dimension (Rate Coding)
    if train_features.ndim > 2:
        print("Flattening time dimension via Mean (Rate Coding)...")
        train_features = train_features.mean(axis=1)  # Or reshape(train_features.shape[0], -1)
        id_test_features = id_test_features.mean(axis=1)
        ood_features = ood_features.mean(axis=1)
    print("Applying L2 Normalization...")
    train_features = normalize(train_features, axis=1, norm='l2')
    id_test_features = normalize(id_test_features, axis=1, norm='l2')
    ood_features = normalize(ood_features, axis=1, norm='l2')
    # 2. Calculate scores
    # Score ID test set
    scores_id = compute_knn_scores(train_features, id_test_features, k=k)
    # Score OoD dataset
    scores_ood = compute_knn_scores(train_features, ood_features, k=k)

    # 3. Evaluation metrics
    # Labels: ID=0, OoD=1
    y_true = np.concatenate([np.zeros(len(scores_id)), np.ones(len(scores_ood))])
    y_scores = np.concatenate([scores_id, scores_ood])  # Larger distance, more likely OoD

    auroc = roc_auc_score(y_true, y_scores)
    aupr = average_precision_score(y_true, y_scores)

    # FPR95 Calculation
    sorted_id_scores = np.sort(scores_id)
    threshold = sorted_id_scores[int(len(sorted_id_scores) * 0.95)]  # 95% ID sample distance threshold
    fpr95 = np.sum(scores_ood < threshold) / len(scores_ood)

    print("\n" + "=" * 40)
    print(f"KNN Result (k={k})")
    print("=" * 40)
    print(f"  AUROC: {auroc * 100:.2f}%")
    print(f"  AUPR:  {aupr * 100:.2f}%")
    print(f"  FPR95: {fpr95 * 100:.2f}%")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.hist(scores_id, bins=50, alpha=0.5, color='green', density=True, label='Test (ID)')
    plt.hist(scores_ood, bins=50, alpha=0.5, color='orange', density=True, label='OoD')
    plt.title(f'KNN Distance Score Distribution (k={k})')
    plt.legend()
    plt.show()


# ==========================================
# 2. Core: Extract features (fix dimension issue)
# ==========================================
def get_spikes_and_preds(model, loader, device):
    model.eval()
    all_spikes = []
    all_preds = []
    all_labels = []

    print(f"Extracting features (samples: {len(loader.dataset)})...")

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Extracting"):
            x = x.to(device, dtype=torch.float32)
            # Key fix: DVS data is usually [B, T, C, H, W]
            # SNN needs [T, B, C, H, W]
            x = x.transpose(0, 1)

            logits, spikes = model(x, return_spike=True)

            preds = logits.argmax(1)

            all_spikes.append(spikes.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.numpy())

            functional.reset_net(model)

    return np.concatenate(all_spikes), np.concatenate(all_preds), np.concatenate(all_labels)


# ==========================================
# 3. Evaluation and Visualization
# ==========================================
def evaluate_metrics(scores_id, scores_ood):
    y_true = np.concatenate([np.zeros(len(scores_id)), np.ones(len(scores_ood))])
    y_scores = np.concatenate([scores_id, scores_ood])

    auroc = roc_auc_score(y_true, y_scores)
    aupr = average_precision_score(y_true, y_scores)  # OoD as Positive

    # FPR95
    sorted_id = np.sort(scores_id)
    threshold = sorted_id[int(len(sorted_id) * 0.95)]  # 95% ID samples have distance smaller than this
    fpr95 = np.sum(scores_ood < threshold) / len(scores_ood)

    print("\n" + "=" * 40)
    print("Graph-Structure SCP Results")
    print("=" * 40)
    print(f"AUROC: {auroc * 100:.2f}%")
    print(f"AUPR:  {aupr * 100:.2f}% (OoD=1)")
    print(f"FPR95: {fpr95 * 100:.2f}%")


def plot_graph_tsne(features_id, features_ood):
    print("\nGenerating t-SNE for Graph Structures...")
    n_samples = 1000
    # Random sampling
    idx_id = np.random.permutation(len(features_id))[:n_samples]
    idx_ood = np.random.permutation(len(features_ood))[:n_samples]

    # Concatenate data
    X = np.concatenate([features_id[idx_id], features_ood[idx_ood]], axis=0)
    y = ['ID (Gesture)'] * len(idx_id) + ['OoD (CIFAR10)'] * len(idx_ood)

    # t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, init='pca', learning_rate=200.0, random_state=42)
    X_emb = tsne.fit_transform(X)

    # Plot
    plt.figure(figsize=(8, 8))
    df = pd.DataFrame({'x': X_emb[:, 0], 'y': X_emb[:, 1], 'Label': y})

    # legend=False removes legend
    sns.scatterplot(data=df, x='x', y='y', hue='Label', style='Label', palette='viridis', legend=False)
    # plt.axis('off') removes x-axis and y-axis
    plt.axis('off')

    plt.savefig('KNN_tsne.pdf', dpi=300)
    plt.show()


# ==========================================
# Main Program
# ==========================================
if __name__ == "__main__":
    import random

    _seed_ = 2020
    random.seed(2020)
    torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
    torch.cuda.manual_seed_all(_seed_)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"

    DVS_GESTURE_PATH = "datasets/DVS128Gesture"
    CIFAR_DVS_PATH = "datasets/CIFAR10DVS"
    MODEL_PATH = 'GC+RGA/CogniSNN-main/GCP/DVSGesture/best_model_ablation.pth'
    NCALTECH_PATH = "datasets/NCaltech101"
    BATCH_SIZE = 8
    FRAMES = 5  # Frames here must correspond to Model's T
    OoD_DATASET = "DVS-Lip"  # "DVS-Lip"  # "N-Caltech101" # "CIFAR10-DVS"  "Corrupted"

    # 1. Model loading
    # Gesture has 11 classes, input channels 2
    model = Model(node_num=5, in_channels=2, out_channels=32, num_classes=11).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model weights loaded.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        sys.exit()

    # 3. Data loading (strictly following your requirements)
    print("Loading Datasets...")

    train_set = DVS128Gesture(root=DVS_GESTURE_PATH, train=True, data_type='frame', frames_number=FRAMES,
                              split_by='number')
    test_set = DVS128Gesture(root=DVS_GESTURE_PATH, train=False, data_type='frame', frames_number=FRAMES,
                             split_by='number')

    train_sampler = torch.utils.data.RandomSampler(train_set)
    # Random sampler. It just returns a random order of batch data
    test_sampler = torch.utils.data.SequentialSampler(test_set)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=8,
        sampler=train_sampler, num_workers=0, pin_memory=True)

    ind_loader = torch.utils.data.DataLoader(
        test_set, batch_size=8,
        sampler=test_sampler, num_workers=0, pin_memory=True)

    model.eval()
    test_samples = 0
    test_acc = 0.
    with torch.no_grad():
        for data in tqdm(ind_loader, desc="evaluation", mininterval=1):
            frame, label = data
            frame = frame.to(device, dtype=torch.float32)
            frame = frame.transpose(0, 1)
            label = label.to(device)
            output = model(frame)
            test_acc += (output.argmax(1) == label).float().sum().item()
            test_samples += label.numel()
            functional.reset_net(model)
        test_acc /= test_samples
        print(test_acc)
    # print(net)
    size = 128
    # print(net)
    train_transform = transforms.Compose([
        lambda x: torch.tensor(x, dtype=torch.float),
        lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
    ])

    if OoD_DATASET == "CIFAR10-DVS":
        # OoD: CIFAR10-DVS (Whole set as OoD)
        ood_set = CIFAR10DVS(root=CIFAR_DVS_PATH, data_type='frame', frames_number=FRAMES, split_by='number')
        # Simple split, take only part for testing to save time
        ood_subset, _ = split_to_train_test_set(0.1, ood_set, 10)
        ood_loader = DataLoader(ood_subset, batch_size=8, shuffle=False, num_workers=0)
    elif OoD_DATASET == "N-Caltech101":
        ood_set = NCaltech101(root=NCALTECH_PATH, data_type='frame', frames_number=FRAMES, split_by='number')
        ood_subset, _ = split_to_train_test_set(0.1, ood_set, 101)
        ood_subset = CustomDataset(ood_subset, train_transform)
        ood_loader = DataLoader(ood_subset, batch_size=8, shuffle=False, num_workers=0)
    elif OoD_DATASET == "DVS-Lip":
        import DVSLip

        test_data_root = 'datasets/DVS-Lip/test'
        training_words = DVSLip.get_training_words()
        label_dct = {k: i for i, k in enumerate(training_words)}
        test_dataset = DVSLip.DVSLipDataset(test_data_root, label_dct, train=False, augment_spatial=False,
                                            augment_temporal=False, T=FRAMES)
        ood_loader = DataLoader(test_dataset, batch_size=8, num_workers=0, shuffle=False, pin_memory=True)

    elif OoD_DATASET == "Corrupted":
        print("\n>>> [Dataset modification] Generating highly corrupted OoD data (Jigsaw + Noise + Shuffle)...")

        # Define "combo" corruption
        transform_severe_corrupt = transforms.Compose([
            lambda x: torch.tensor(x, dtype=torch.float),  # 1. Basic conversion
            lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),  # 2. Resize

            # --- High energy warning: corruption begins ---
            SpatialJigsaw(grid_size=2),  # 3. Spatial fragmentation (destroy graph topology)
            TemporalShuffle(),  # 4. Temporal shuffling (destroy motion coherence)
            AddGaussianNoise(std=0.5),  # 5. Inject noise (destroy spike purity)
            # ------------------------
        ])

        # Use DVS128Gesture test set as baseplate, apply corruption
        # Note: Must re-instantiate dataset to apply new transform
        ood_set = DVS128Gesture(
            root=DVS_GESTURE_PATH,
            train=False,  # Use test set as baseplate
            data_type='frame',
            frames_number=FRAMES,
            split_by='number',
            transform=transform_severe_corrupt  # <--- Inject "poison"
        )

        ood_loader = DataLoader(ood_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    spikes_train, _, labels_train = get_spikes_and_preds(model, train_loader, device)
    spikes_test, preds_test, _ = get_spikes_and_preds(model, ind_loader, device)
    spikes_ood, preds_ood, _ = get_spikes_and_preds(model, ood_loader, device)

    # -------------------------------------------------
    # Replace or add KNN analysis
    # -------------------------------------------------

    # Parameter K selection:
    # Generally, larger sample size allows larger k.
    # For smaller datasets like DVSGesture, k=5, 10, 50 are common trial values.
    # Paper "Deep k-Nearest Neighbors for Out-of-Distribution Detection" recommends k=50 (ImageNet scale),
    # Smaller datasets may work better with k=5 or k=10.

    run_knn_analysis(spikes_train, spikes_test, spikes_ood, k=5)

    # If you want to compare k's impact, you can run in a loop:
    # for k in [1, 5, 10, 50]:
    #     run_knn_analysis(spikes_train, spikes_test, spikes_ood, k=k)

    # t-SNE can continue to be drawn
    plot_graph_tsne(spikes_test, spikes_ood)