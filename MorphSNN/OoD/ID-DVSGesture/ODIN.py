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
import scipy.special as scysp
# 引入你的模型定义
from smodel import Model

# ==========================================
# 1. t-SNE 可视化
# ==========================================
def plot_tsne(features_id, features_ood, max_samples=1000, title="t-SNE Visualization"):
    print(f"\n>>> 正在生成 t-SNE 图 (采样数: {max_samples})...")

    n_id = min(len(features_id), max_samples)
    n_ood = min(len(features_ood), max_samples)

    idx_id = np.random.permutation(len(features_id))[:n_id]
    idx_ood = np.random.permutation(len(features_ood))[:n_ood]

    data_id = features_id[idx_id]
    data_ood = features_ood[idx_ood]

    X = np.concatenate([data_id, data_ood], axis=0)
    labels = ['ID (Gesture)'] * len(data_id) + ['OoD (CIFAR10)'] * len(data_ood)

    tsne = TSNE(n_components=2, init='pca', learning_rate=200.0, random_state=42, n_jobs=-1)
    X_embedded = tsne.fit_transform(X)

    plt.figure(figsize=(8, 8))
    df = pd.DataFrame({'x': X_embedded[:, 0], 'y': X_embedded[:, 1], 'Label': labels})
    sns.scatterplot(data=df, x='x', y='y', hue='Label', style='Label',
                    palette={'ID (Gesture)': 'dodgerblue', 'OoD (CIFAR10)': 'darkorange'},
                    alpha=0.7, s=50)

    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.savefig('tsne_odin_dvs.png', dpi=300)
    plt.show()


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




# 2. 提取 Logits (针对 DVS 数据修正)
# ==========================================
def get_logits(model, loader, device):
    model.eval()
    all_logits = []
    print(f"正在提取 Logits (样本数: {len(loader.dataset)})...")

    with torch.no_grad():
        for x, _ in tqdm(loader, desc="Extracting"):
            x = x.to(device, dtype=torch.float32)
            # DVS 关键操作: [N, T, C, H, W] -> [T, N, C, H, W]
            x = x.transpose(0, 1)
            logits = model(x)
            all_logits.append(logits.cpu().numpy())
            functional.reset_net(model)

    return np.concatenate(all_logits)


def run_odin_analysis(logits_train, logits_test, logits_ood):
    print("\n>>> 开始执行 ODIN 算法 (自动搜索最佳温度 T)...")

    temperatures = [1, 10, 100, 1000]
    best_auroc = 0
    best_results = {}
    best_scores = {}
    best_scaled_logits = {}
    best_temp = 1

    for temp in temperatures:
        # 1. Temperature Scaling
        prob_train = scysp.softmax(logits_train / temp, axis=1)
        prob_test = scysp.softmax(logits_test / temp, axis=1)
        prob_ood = scysp.softmax(logits_ood / temp, axis=1)

        # 2. Max Softmax Score (Confidence)
        # 分数越大 -> 越确信是 ID
        scores_train = np.max(prob_train, axis=1)
        scores_test = np.max(prob_test, axis=1)
        scores_ood = np.max(prob_ood, axis=1)

        # --- Metrics 计算 ---
        
        # 准备数据
        y_scores = np.concatenate([scores_test, scores_ood])
        
        # (1) AUROC: 方向无关 (ID=1, OoD=0)
        y_true = np.concatenate([np.ones(len(scores_test)), np.zeros(len(scores_ood))])
        auroc = roc_auc_score(y_true, y_scores)

        # (2) AUPR-In (ID as Positive)
        # Baseline ~ 3% (288 / 9288)
        aupr_in = average_precision_score(y_true, y_scores)

        # (3) AUPR-Out (OoD as Positive)
        # Baseline ~ 97% (9000 / 9288)
        # 注意: ODIN 分数越低越是 OoD，所以取负号
        y_true_out = np.concatenate([np.zeros(len(scores_test)), np.ones(len(scores_ood))])
        aupr_out = average_precision_score(y_true_out, -y_scores)

        # (4) FPR95
        # 阈值: 95% 的 ID 训练样本分数大于此值
        sorted_train = np.sort(scores_train)
        threshold_95 = sorted_train[int(len(sorted_train) * 0.05)]
        # 误报: OoD 样本分数 > 阈值
        fpr95 = np.sum(scores_ood > threshold_95) / len(scores_ood)

        print(f"  [Temp={temp}] AUROC: {auroc * 100:.2f}%, FPR95: {fpr95 * 100:.2f}%")

        # 记录最佳结果 (以 AUROC 为准)
        if auroc > best_auroc:
            best_auroc = auroc
            best_temp = temp
            best_results = {
                'AUROC': auroc, 
                'AUPR_IN': aupr_in, 
                'AUPR_OUT': aupr_out, 
                'FPR95': fpr95
            }
            best_scores = {'Train': scores_train, 'Test': scores_test, 'OoD': scores_ood}
            best_scaled_logits = {'Test': logits_test / temp, 'OoD': logits_ood / temp}

    print("\n" + "=" * 40)
    print(f"ODIN 最终结果 (Gesture vs CIFAR10-DVS, Best Temp={best_temp})")
    print("=" * 40)
    print(f"  AUROC:     {best_results['AUROC'] * 100:.2f}%")
    print(f"  FPR95:     {best_results['FPR95'] * 100:.2f}%")
    print("-" * 20)
    print(f"  AUPR-In:   {best_results['AUPR_IN'] * 100:.2f}% (ID as Positive)")
    print(f"  AUPR-Out:  {best_results['AUPR_OUT'] * 100:.2f}% (OoD as Positive)")

    # 绘图
    plt.figure(figsize=(8, 6))
    plt.hist(best_scores['Train'], bins=50, alpha=0.5, color='blue', density=True, label='Train (ID)')
    plt.hist(best_scores['Test'], bins=50, alpha=0.5, color='green', density=True, label='Test (ID)')
    plt.hist(best_scores['OoD'], bins=50, alpha=0.5, color='orange', density=True, label='OoD (CIFAR10)')
    plt.title(f'ODIN Score Distribution (T={best_temp})')
    plt.xlabel('Confidence Score (Higher is ID)')
    plt.legend()
    plt.show()

    plot_tsne(best_scaled_logits['Test'], best_scaled_logits['OoD'],
              title=f"t-SNE of Scaled Logits (T={best_temp})")




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
    model = Model(node_num=5, in_channels=2, out_channels=32, num_classes=11).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model weights loaded.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        sys.exit()
    print("Loading Datasets...")
    
    train_set = DVS128Gesture(root=DVS_GESTURE_PATH, train=True, data_type='frame', frames_number=FRAMES,
                              split_by='number')
    test_set = DVS128Gesture(root=DVS_GESTURE_PATH, train=False, data_type='frame', frames_number=FRAMES,
                             split_by='number')

    train_sampler = torch.utils.data.RandomSampler(train_set)
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
        
    from spikingjelly.datasets.n_caltech101 import NCaltech101
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

   
    # 4. 提取特征
    logits_train = get_logits(model, train_loader, device)
    logits_test = get_logits(model, ind_loader, device)
    logits_ood = get_logits(model, ood_loader, device)

    run_odin_analysis(logits_train, logits_test, logits_ood)

