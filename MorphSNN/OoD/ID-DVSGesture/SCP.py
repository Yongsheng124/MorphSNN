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
# 引入你的模型定义
from smodel import Model

# ==========================================
# 0. 数据集工具 (保持你的逻辑)
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

def split_to_train_test_set(train_ratio, origin_dataset, num_classes, random_split=False, transform_train=None, transform_test=None):
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




# ==========================================
# 2. 核心：提取特征 (修正维度问题)
# ==========================================
def get_spikes_and_preds(model, loader, device):
    model.eval()
    all_spikes = []
    all_preds = []
    all_labels = []

    print(f"正在提取特征 (样本数: {len(loader.dataset)})...")

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Extracting"):
            x = x.to(device, dtype=torch.float32)
            # 关键修正: DVS数据通常是 [B, T, C, H, W]
            # SNN 需要 [T, B, C, H, W]
            x = x.transpose(0, 1)

            logits, spikes  = model(x, return_spike = True)

            preds = logits.argmax(1)

            all_spikes.append(spikes.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.numpy())

            functional.reset_net(model)

    return np.concatenate(all_spikes), np.concatenate(all_preds), np.concatenate(all_labels)







# ==========================================
# 3. SCP 算法部分
# ==========================================
def compute_archetypes(train_spikes, train_labels, num_classes):
    archetypes = {}
    print(f"\n>>> Phase 1: 构建 SCP 表征 (Total Classes: {num_classes})...")

    for c in range(num_classes):
        class_data = train_spikes[train_labels == c]

        # 如果某个类没有数据（这在 DVS 数据集切分不均时可能发生），跳过或报错
        if len(class_data) == 0:
            print(f"Warning: Class {c} has no samples!")
            archetypes[c] = np.array([])
            continue

        if len(class_data) > 1000:
            indices = np.random.choice(len(class_data), 1000, replace=False)
            class_data = class_data[indices]

        # 层次聚类
        try:
            Z = linkage(class_data, method='ward')
            clusters = fcluster(Z, t=5, criterion='maxclust')

            class_archetypes = []
            for cluster_id in np.unique(clusters):
                centroid = np.median(class_data[clusters == cluster_id], axis=0)
                class_archetypes.append(centroid)
            archetypes[c] = np.array(class_archetypes)
        except Exception as e:
            print(f"Error clustering class {c}: {e}")
            archetypes[c] = np.array([np.median(class_data, axis=0)])  # 聚类失败则用全局中位数兜底

    return archetypes



def compute_scp_scores(spikes, pred_labels, archetypes):
    scores = []
    for i in tqdm(range(len(spikes)), desc="Scoring"):
        sample = spikes[i].reshape(1, -1)
        pred = pred_labels[i]

        if pred not in archetypes or len(archetypes[pred]) == 0:
            scores.append(1e9)  # 异常情况
            continue

        centroids = archetypes[pred]
        dists = cdist(sample, centroids, metric='cityblock')
        scores.append(np.min(dists))

    return np.array(scores)





def run_scp_analysis(scores_train, scores_test, scores_ood):
    print("\n>>> 开始评估 SCP 性能...")

    # ID=0, OoD=1
    y_true = np.concatenate([np.zeros(len(scores_test)), np.ones(len(scores_ood))])
    y_scores = np.concatenate([scores_test, scores_ood])

    auroc = roc_auc_score(y_true, y_scores)
    aupr = average_precision_score(y_true, y_scores)

    sorted_train = np.sort(scores_train)
    threshold_95 = sorted_train[int(len(sorted_train) * 0.95)]
    fpr95 = np.sum(scores_ood < threshold_95) / len(scores_ood)

    print("\n" + "=" * 40)
    print(f"SCP Result (ID: Gesture vs OoD: CIFAR10-DVS)")
    print("=" * 40)
    print(f"  AUROC: {auroc * 100:.2f}%")
    print(f"  AUPR:  {aupr * 100:.2f}%")
    print(f"  FPR95: {fpr95 * 100:.2f}%")

    plt.figure(figsize=(8, 6))
    plt.hist(scores_train, bins=50, alpha=0.5, color='blue', density=True, label='Train (ID)')
    plt.hist(scores_test, bins=50, alpha=0.5, color='green', density=True, label='Test (ID)')
    plt.hist(scores_ood, bins=50, alpha=0.5, color='orange', density=True, label='OoD')
    plt.title('SCP Score Distribution')
    plt.legend()
    plt.show()












# ==========================================
# 3. 评估与可视化
# ==========================================
def evaluate_metrics(scores_id, scores_ood):
    y_true = np.concatenate([np.zeros(len(scores_id)), np.ones(len(scores_ood))])
    y_scores = np.concatenate([scores_id, scores_ood])

    auroc = roc_auc_score(y_true, y_scores)
    aupr = average_precision_score(y_true, y_scores) # OoD as Positive

    # FPR95
    sorted_id = np.sort(scores_id)
    threshold = sorted_id[int(len(sorted_id) * 0.95)] # 95% ID 样本距离小于此
    fpr95 = np.sum(scores_ood < threshold) / len(scores_ood)

    print("\n" + "="*40)
    print("Graph-Structure SCP Results")
    print("="*40)
    print(f"AUROC: {auroc*100:.2f}%")
    print(f"AUPR:  {aupr*100:.2f}% (OoD=1)")
    print(f"FPR95: {fpr95*100:.2f}%")

def plot_graph_tsne(features_id, features_ood, title="t-SNE of Graph Structures"):
    print("\nGenerating t-SNE for Graph Structures...")
    n_samples = 1000
    idx_id = np.random.permutation(len(features_id))[:n_samples]
    idx_ood = np.random.permutation(len(features_ood))[:n_samples]
    
    X = np.concatenate([features_id[idx_id], features_ood[idx_ood]], axis=0)
    y = ['ID (Gesture)'] * len(idx_id) + ['OoD (CIFAR10)'] * len(idx_ood)
    
    tsne = TSNE(n_components=2, init='pca', learning_rate=200.0, random_state=42)
    X_emb = tsne.fit_transform(X)
    
    plt.figure(figsize=(8,8))
    df = pd.DataFrame({'x': X_emb[:,0], 'y': X_emb[:,1], 'Label': y})
    sns.scatterplot(data=df, x='x', y='y', hue='Label', style='Label', palette='viridis')
    plt.title(title)
    plt.savefig('scp_tsne.png', dpi=300)
    plt.show()

# ==========================================
# 主程序
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
    
    DVS_GESTURE_PATH = "/home/hys/datasets/DVS128Gesture"
    CIFAR_DVS_PATH = "/home/hys/datasets/CIFAR10DVS"
    MODEL_PATH = '/home/hys/GC+RGA/CogniSNN-main/GCP/DVSGesture/best_model_ablation.pth'
    NCALTECH_PATH  =  "/home/hys/datasets/NCaltech101"
    BATCH_SIZE = 8
    FRAMES = 5  # 这里的 Frames 必须对应 Model 的 T
    OoD_DATASET = "DVS-Lip"   #  "DVS-Lip"  # "N-Caltech101" # "CIFAR10-DVS"

    # 1. 模型加载
    # Gesture 有 11 类，输入通道 2
    model = Model(node_num=5, in_channels=2, out_channels=32, num_classes=11).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model weights loaded.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        sys.exit()

    # 3. 数据加载 (严格按照你要求的设置)
    print("Loading Datasets...")
    
    train_set = DVS128Gesture(root=DVS_GESTURE_PATH, train=True, data_type='frame', frames_number=FRAMES,
                              split_by='number')
    test_set = DVS128Gesture(root=DVS_GESTURE_PATH, train=False, data_type='frame', frames_number=FRAMES,
                             split_by='number')

    train_sampler = torch.utils.data.RandomSampler(train_set)
    # 随机采样器。可以就是返回一个随机顺序的批次数据
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
            output= model(frame)
            test_acc += (output.argmax(1) == label).float().sum().item()
            test_samples += label.numel()
            functional.reset_net(model)
        test_acc /= test_samples
        print(test_acc)

    if OoD_DATASET == "CIFAR10-DVS":
        # OoD: CIFAR10-DVS (Whole set as OoD)
        ood_set = CIFAR10DVS(root=CIFAR_DVS_PATH, data_type='frame', frames_number=FRAMES, split_by='number')
    # 简单切分，只取一部分做测试以节省时间
        ood_subset, _  = split_to_train_test_set(0.1, ood_set, 10)
        ood_loader = DataLoader(ood_subset, batch_size=8, shuffle=False, num_workers=0)
    elif OoD_DATASET == "N-Caltech101":
        ood_set = NCaltech101(root=NCALTECH_PATH , data_type='frame', frames_number=FRAMES, split_by='number')
        ood_subset, _  = split_to_train_test_set(0.1, ood_set, 101)
        ood_loader = DataLoader(ood_subset, batch_size=8, shuffle=False, num_workers=0)
    elif OoD_DATASET == "DVS-Lip":
        import DVSLip
        test_data_root = '/home/hys/datasets/DVS-Lip/test'
        training_words = DVSLip.get_training_words()
        label_dct = {k: i for i, k in enumerate(training_words)}
        test_dataset = DVSLip.DVSLipDataset(test_data_root, label_dct, train=False, augment_spatial=False,
                                        augment_temporal=False, T=FRAMES)
        ood_loader = DataLoader(test_dataset, batch_size=8, num_workers=0, shuffle=False, pin_memory=True)



    spikes_train, _, labels_train = get_spikes_and_preds(model, train_loader, device)


    spikes_test, preds_test, _ = get_spikes_and_preds(model, ind_loader, device)


    spikes_ood, preds_ood, _ = get_spikes_and_preds(model, ood_loader, device)


    NUM_CLASSES_ID = 11

    archetypes = compute_archetypes(spikes_train, labels_train, num_classes=NUM_CLASSES_ID)

    scores_train = compute_scp_scores(spikes_train, labels_train, archetypes)
    scores_test = compute_scp_scores(spikes_test, preds_test, archetypes)
    scores_ood = compute_scp_scores(spikes_ood, preds_ood, archetypes)

    run_scp_analysis(scores_train, scores_test, scores_ood)

    # t-SNE
    plot_graph_tsne( spikes_test, spikes_ood)