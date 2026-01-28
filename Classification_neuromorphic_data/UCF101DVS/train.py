from torch.utils.data import DataLoader
# 引入上面写的类
from dataset import UCF101DVS_ProcessedDataset
from spikingjelly.datasets import play_frame
import torch
import numpy as np
import math
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

import os
import glob
import random
from torch.utils.data import Dataset, DataLoader
import torch
from spikingjelly.clock_driven import functional
from spikingjelly.datasets.n_caltech101 import NCaltech101
import matplotlib.pyplot as plt
import os
from torch.cuda import amp

_seed_ = 2020
import random
import math

random.seed(2020)
torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
import json

np.random.seed(_seed_)

DATA_DIR = "datasets/UCF101DVS/UCF101DVS_ProcessedT10"
BATCH_SIZE = 32
NUM_WORKERS = 0


# ===========================================


def get_ucf101_split_files(root_dir, test_groups=None, extension='.pt'):
    if test_groups is None:
        test_groups = [20, 21, 22, 23, 24, 25]

    train_files = []
    test_files = []

    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    print(f"找到 {len(classes)} 个类别。正在根据 Group 划分数据...")

    for target_class in classes:
        class_dir = os.path.join(root_dir, target_class)

        files = glob.glob(os.path.join(class_dir, f"**/*{extension}"), recursive=True)

        for file_path in files:
            file_name = os.path.basename(file_path)

            try:

                parts = file_name.split('_')
                group_part = [p for p in parts if p.startswith('g') and p[1:].isdigit()]

                if not group_part:
                    print(f"警告: 跳过无法解析 Group 的文件: {file_name}")
                    continue

                group_num = int(group_part[0][1:])

                sample = (file_path, class_to_idx[target_class])

                if group_num in test_groups:
                    test_files.append(sample)
                else:
                    train_files.append(sample)

            except Exception as e:
                print(f"解析错误 {file_name}: {e}")

    random.shuffle(train_files)

    print(f"划分完成!")
    print(f"训练集样本数: {len(train_files)} (Group 01-19)")
    print(f"测试集样本数: {len(test_files)}  (Group {test_groups})")

    return train_files, test_files, class_to_idx


class UCF101_Split_Dataset(Dataset):
    def __init__(self, file_list, transform=None):

        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path, label = self.file_list[idx]

        try:
            data = torch.load(file_path)
        except Exception as e:
            # 处理坏文件的情况
            print(f"Error loading {file_path}: {e}")
            # 简单的容错：随机返回另一个样本，或者报错
            return self.__getitem__(random.randint(0, len(self.file_list) - 1))

        if self.transform:
            data = self.transform(data)

        return data, label


def main():
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--T', type=int, default=10, help='time step of neuron, (default: 5)')
    parser.add_argument('--epochs', type=int, default=192, help='number of epochs, (default: 100)')
    parser.add_argument('--p', type=float, default=0.75, help='graph probability, (default: 0.75)')
    parser.add_argument('--c', type=int, default=109,
                        help='channel count for each node, (example: 78, 109, 154), (default: 154)')
    parser.add_argument('--k', type=int, default=4,
                        help='each node is connected to k nearest neighbors in ring topology, (default: 4)')
    parser.add_argument('--m', type=int, default=3,
                        help='number of edges to attach from a new node to existing nodes, (default: 5)')
    parser.add_argument('--graph-mode', type=str, default="ER",
                        help="random graph, (Example: ER, WS, BA), (default: ER)")
    parser.add_argument('--node-num', type=int, default=5, help="Number o f graph node (default n=32)")
    parser.add_argument('--learning-rate', type=float, default=0.001, help='learning rate, (default: 1e-1)')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size, (default: 100)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='Momentum for SGD. Adam will not use momentum')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 0)',
                        dest='weight_decay')
    parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--data-dir', default="datasets/NCaltech101", type=str)

    args = parser.parse_args()

    config_and_results = {}

    device = args.device

    output_dir = "./" + str(args.graph_mode) + "(" + str(args.node_num) + "," + str(args.p) + ")" + "C" + str(
        args.c) + "T" + str(args.T) + "lapician"

    os.makedirs(output_dir, exist_ok=True)

    # from smodelT import Model

    from smodel import Model
    net = Model(node_num=7, in_channels=2, out_channels=156, num_classes=101).to(device)

    size = 64

    train_transform = transforms.Compose([
        lambda x: torch.tensor(x, dtype=torch.float),
        lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
        transforms.RandomCrop(size, padding=size // 12),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15)

    ])

    test_transform = transforms.Compose([
        lambda x: torch.tensor(x, dtype=torch.float),
        lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
    ])

    train_files, test_files, class_to_idx = get_ucf101_split_files(
        root_dir=DATA_DIR,
        test_groups=[23, 24, 25],
        extension='.pt'
    )

    print(f"实际划分结果：")
    print(f"训练集 (Groups 01-22): {len(train_files)} 个样本")
    print(f"测试集 (Groups 23-25): {len(test_files)} 个样本")
    if len(train_files) + len(test_files) > 0:
        print(
            f"实际比例约: {len(train_files) / (len(train_files) + len(test_files)):.2%} : {len(test_files) / (len(train_files) + len(test_files)):.2%}")

    train_set = UCF101_Split_Dataset(train_files, transform=train_transform)
    test_set = UCF101_Split_Dataset(test_files, transform=test_transform)

    train_sampler = torch.utils.data.RandomSampler(train_set)

    test_sampler = torch.utils.data.SequentialSampler(test_set)

    train_data_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=0, pin_memory=True)

    test_data_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=0, pin_memory=True)

    loss_fn = nn.CrossEntropyLoss().to(device)

    epochs = args.epochs
    # 5e-4
    optimizer = torch.optim.SGD(
        net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=64, gamma=0.1)

    use_amp = True
    if use_amp:
        print("使用了混合精度训练")
        scaler = amp.GradScaler()
    else:
        scaler = None

    epoch_list = []
    test_acc_list = []
    train_acc_list = []
    train_loss_list = []
    test_loss_list = []
    max_test_acc = 0.

    x = torch.round(torch.rand((args.T, 1, 2, size, size))).to(args.device)
    Flops, params = profile(net, inputs=(x,))
    print('Flops: %.4fG' % (Flops / 1e9))  # 将 FLOPs 转换为 GFLOPs
    print('params参数量: % .4fM' % (params / 1000000))

    time1 = time.time()
    for epoch in range(epochs):
        epoch_list.append(epoch + 1)
        start_time = time.time()
        net.train()
        step = 0
        train_loss = 0
        train_acc = 0
        train_samples = 0
        print("---------第{}轮训练开始--------".format(epoch + 1))
        for data in tqdm(train_data_loader, desc="epoch " + str(epoch + 1), mininterval=1):
            optimizer.zero_grad()
            img, label = data
            img = img.to(device)
            img = img.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]

            label = label.to(device)
            label_onehot = F.one_hot(label, 101).float()
            if use_amp:
                with amp.autocast():
                    output = net(img)
                    loss = loss_fn(output, label_onehot)
            else:
                output = net(img)
                loss = loss_fn(output, label_onehot)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            functional.reset_net(net)

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (output.argmax(1) == label).float().sum().item()
            step += 1
            if step % 100 == 0:
                print("[Epoch {0:4d}] Loss: {1:2.3f} Acc: {2:.3f}%".format(epoch + 1, loss.data,
                                                                           (train_acc * 100 / train_samples)), end='')
                for param_group in optimizer.param_groups:
                    print(",  Current learning rate is: {}".format(param_group['lr']))

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples
        print(
            "Epoch{}: Train_acc {}; Train_loss {}; Time of train {}; Speed of train {};".format(
                epoch + 1, train_acc, train_loss,
                (train_time - start_time), train_speed))
        for param_group in optimizer.param_groups:
            print("Current learning rate is: {}".format(param_group['lr']))
        lr_scheduler.step()

        net.eval()
        test_samples = 0
        test_acc = 0.
        test_loss = 0.
        with torch.no_grad():
            for data in tqdm(test_data_loader, desc="evaluation", mininterval=1):
                frame, label = data
                frame = frame.to(device)
                frame = frame.transpose(0, 1)

                label = label.to(device)
                label_onehot = F.one_hot(label, 101).float()
                output = net(frame)
                loss = loss_fn(output, label_onehot)
                test_loss += loss.item() * label.numel()
                test_acc += (output.argmax(1) == label).float().sum().item()
                test_samples += label.numel()
                functional.reset_net(net)

            test_acc /= test_samples
            test_loss /= test_samples
            test_acc_list.append(test_acc)
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            test_loss_list.append(test_loss)
            print('Epoch{}: Test set accuracy: {:.5f}, Best accuracy: {:.5f}'.format(epoch + 1, test_acc,
                                                                                     max_test_acc))
            if max_test_acc < test_acc:
                max_test_acc = test_acc
                torch.save(net.state_dict(), output_dir + "/best_model.pth")

    time2 = time.time()
    train_time = time2 - time1

    hours = int(train_time / 3600)
    minutes = int((train_time % 3600) / 60)
    seconds = int(train_time % 60)

    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    config_and_results['train_time'] = time_str
    config_and_results['max_test_acc'] = max_test_acc
    config_and_results['train_loss_list'] = train_loss_list
    config_and_results['train_acc_list'] = train_acc_list
    config_and_results['test_loss_list'] = test_loss_list
    config_and_results['test_acc_list'] = test_acc_list

    print("train_time:", time_str)

    with open(output_dir + '/config_and_results.txt', 'a') as file:
        for key, value in config_and_results.items():
            dict_str = json.dumps({key: value}, indent=4)
            file.write(dict_str + '\n')


if __name__ == "__main__":
    main()
