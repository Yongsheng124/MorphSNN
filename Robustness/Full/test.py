import argparse
import os.path
import pickle
import time
import torch.nn.functional as F
import torch
from thop import profile
from torch import nn
from torchvision import transforms
from tqdm import tqdm

from model5 import Model
from spikingjelly.clock_driven import functional
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
import matplotlib.pyplot as plt
import os
from torch.cuda import amp
import sys
_seed_ = 2020
import random

random.seed(_seed_)
torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def _to_BTCHW(x: torch.Tensor):
    """接受 [B,T,C,H,W] 或 [T,B,C,H,W]，统一成 [B,T,C,H,W] 并返回 (x5d, layout)"""
    if x.dim() != 5:
        raise ValueError("x must be 5D: [B,T,C,H,W] or [T,B,C,H,W]")
    if x.shape[0] < 8 and x.shape[1] > x.shape[0]:   # 粗略判断 [T,B,...]
        # [T,B,C,H,W] -> [B,T,C,H,W]
        return x.permute(1,0,2,3,4).contiguous(), "TBCHW"
    return x.contiguous(), "BTCHW"

def _restore_layout(y: torch.Tensor, layout: str):
    return y.permute(1,0,2,3,4).contiguous() if layout == "TBCHW" else y

@torch.no_grad()
def row_drop_ratio(x: torch.Tensor, ratio: float = 0.1, share_across_time: bool = True) -> torch.Tensor:
    """
    按比例丢失若干行（整条扫描线置零）。
    ratio: 丢失行数 / 总行数（0~1）
    share_across_time: True=同一掩码应用到所有时间步（推荐，模拟硬件缺行）
    """
    x5d, layout = _to_BTCHW(x)
    B, T, C, H, W = x5d.shape
    k = max(0, min(H, int(round(ratio * H))))   # 要丢的行数
    if k == 0:
        return x

    y = x5d.clone()
    if share_across_time:
        # 每个样本一张行掩码，整段时间共享
        for b in range(B):
            rows = torch.randperm(H, device=x5d.device)[:k]
            y[b, :, :, rows, :] = 0
    else:
        # 每个样本、每个时间步各自抽一组行
        for b in range(B):
            for t in range(T):
                rows = torch.randperm(H, device=x5d.device)[:k]
                y[b, t, :, rows, :] = 0

    return _restore_layout(y, layout)

def add_sp(x, eps):
    """Salt‑&‑Pepper: ε×0.02 flip ratio."""
    p = eps * 0.02
    m = torch.rand_like(x)
    flip = m < p
    x_flip = x.clone()
    x_flip[flip] = 1. - x_flip[flip]
    return x_flip

def add_pn(x, eps: float):

    lam = torch.clamp(x * eps, min=0.0)

    # 2. 采样泊松随机数并做零均值化
    noise = torch.poisson(lam) - lam

    # 3. 叠加噪声并返回
    return x + noise

# location = "best_model2.pth"
#
# location = "reproduce_res_growing.pth"

model = Model(5,2,32).to("cuda:0")
location = "STSP/Robustness/dynamic/best_model_ablation.pth"
model.load_state_dict(torch.load(location, map_location="cuda:0")) 


model.eval()
device = "cuda:0"
model = model.to(device)
model.eval()
test_set = DVS128Gesture(root="datasets/DVS128Gesture", train=False, data_type='frame', frames_number=5,
                         split_by='number')

test_sampler = torch.utils.data.SequentialSampler(test_set)

test_data_loader = torch.utils.data.DataLoader(
    test_set, batch_size= 8,
    sampler=test_sampler, num_workers=0, pin_memory=True)
# 在gpu上训练的时候使用pin_memory = True能够保证数据存储在固定内存上更快速的训练。
model.eval()
# model.transfrom_mode("test")
print(model)
device = 'cuda'
res = []
for q in range(10):
    test_acc = 0.
    test_samples = 0
    for data in tqdm(test_data_loader, desc="evaluation", mininterval=1):
        frame, label = data
        frame = frame.to(device, dtype=torch.float32)
        frame = frame.transpose(0, 1)

        #  测试鲁棒性
        # frame = add_sp(frame, q)

        # frame = add_pn(frame, q)


        frame = row_drop_ratio(frame, ratio=q*0.05, share_across_time=True)
        label = label.to(device)
        output = model(frame)
        # sys.exit()
        test_acc += (output.argmax(1) == label).float().sum().item()
        test_samples += label.numel()
        functional.reset_net(model)
    test_acc /= test_samples
    res.append(round(test_acc, 4))
print(res)


