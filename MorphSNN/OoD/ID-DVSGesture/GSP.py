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
# 1. 核心: 提取图结构特征 (Graph Structure S)
# ==========================================
def get_graph_features(model, loader, device):
    """
    提取每个样本的动态图结构 S。
    返回:
        features: [Samples, T * N * N] 的展平向量
        labels: [Samples]
    """
    model.eval()
    all_graphs = []
    all_labels = []

    print(f"提取图结构特征 S (样本数: {len(loader.dataset)})...")

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Extracting Graphs"):
            x = x.to(device, dtype=torch.float32)
            # DVS Transpose: [B, T, C, H, W] -> [T, B, C, H, W]
            x = x.transpose(0, 1)
    
            # 返回值: (logits, graph_tensor)
            # graph_tensor shape: [B, T, N, N]
            _, graph_tensor = model(x, return_graph=True)

            # 将图结构展平作为特征向量
            # [B, T, N, N] -> [B, T * N * N]
            B = graph_tensor.shape[0]
            flat_graph = graph_tensor.reshape(B, 5*7*7)

            all_graphs.append(flat_graph.cpu().numpy())
            all_labels.append(y.numpy())

            functional.reset_net(model)

    return np.concatenate(all_graphs), np.concatenate(all_labels)

# ==========================================
# 2. SCP (Graph Prototypes)
# ==========================================
def compute_graph_prototypes(train_graphs, train_labels, num_classes=11):
    """
    计算每个类别的图模式中心 (Graph Prototype)
    """
    prototypes = {}
    print("\n计算图结构原型 (Prototypes)...")
    
    for c in range(num_classes):
        # 获取该类的所有图向量
        class_data = train_graphs[train_labels == c]
        
        if len(class_data) == 0:
            print(f"Warning: Class {c} is empty.")
            prototypes[c] = np.zeros((1, train_graphs.shape[1]))
            continue
            
        # 计算均值作为该类的"标准图模式"
        # 也可以尝试 Median 或 K-Means 聚类
        centroid = np.mean(class_data, axis=0).reshape(1, -1)
        prototypes[c] = centroid
        
    return prototypes

def calculate_ood_scores(features, prototypes):
    """
    计算每个样本到最近图原型的距离
    距离越大 -> 越不符合已知图模式 -> OoD
    """
    scores = []
    num_classes = len(prototypes)
    
    # 将所有原型堆叠成矩阵 [Num_Classes, Feat_Dim]
    proto_matrix = np.vstack([prototypes[c] for c in range(num_classes)])
    
    # 批量计算距离 (使用 L2 距离，也可以尝试 'cityblock' L1)
    # dists: [N_samples, Num_Classes]
    dists = cdist(features, proto_matrix, metric='euclidean')
    
    # 取到最近类别的距离作为分数
    min_dists = np.min(dists, axis=1)
    
    return min_dists

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
    plt.savefig('graph_tsne.png', dpi=300)
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
    
    DVS_GESTURE_PATH = "E:\\datasets\\DVS128Gesture"
    CIFAR_DVS_PATH = "E:\\datasets\\CIFAR10-DVS"
    MODEL_PATH = 'best_model_ablation.pth'
    NCALTECH_PATH  =  "/home/hys/datasets/NCaltech101"
    BATCH_SIZE = 8
    FRAMES = 5  # 这里的 Frames 必须对应 Model 的 T
    OoD_DATASET = "CIFAR10-DVS"   #  "DVS-Lip"  # "N-Caltech101" # "CIFAR10-DVS"

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



    # 4. 提取图特征
    # graph_train shape: [N_train, T*N*N]
    graph_train, labels_train = get_graph_features(model, train_loader, device)
    graph_test, _ = get_graph_features(model, ind_loader, device)
    graph_ood, _ = get_graph_features(model, ood_loader, device)

    # 5. SCP 流程 (基于 Graph Structure)
    # 计算 ID 训练集的图原型
    # 注意: Gesture 有 11 类 (0-10)
    prototypes = compute_graph_prototypes(graph_train, labels_train, num_classes=11)

    # 计算距离分数
    scores_test = calculate_ood_scores(graph_test, prototypes)
    scores_ood = calculate_ood_scores(graph_ood, prototypes)

    # 6. 结果与绘图
    evaluate_metrics(scores_test, scores_ood)
    
    # 绘制图结构的直方图
    plt.figure(figsize=(8,6))
    plt.hist(scores_test, bins=50, alpha=0.5, label='ID (Gesture)', density=True)
    plt.hist(scores_ood, bins=50, alpha=0.5, label='OoD (CIFAR10)', density=True)
    plt.title("Distribution of Graph Structure Distances")
    plt.xlabel("L2 Distance to Nearest Graph Prototype")
    plt.legend()
    plt.show()

    # t-SNE
    plot_graph_tsne(graph_test, graph_ood)