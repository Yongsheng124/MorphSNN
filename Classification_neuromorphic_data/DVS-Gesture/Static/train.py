import argparse
import os.path
import pickle
import time
import torch.nn.functional as F
import torch
from thop import profile
from torch import nn
from tqdm import tqdm
from spikingjelly.clock_driven import functional
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
import matplotlib.pyplot as plt
import os
from torch.cuda import amp
import datetime

_seed_ = 2020
import random

random.seed(2020)
torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
import json
import math

np.random.seed(_seed_)


def save_res_list(result, name, output_dir):
    if os.path.isdir(output_dir + '/list/'):
        with open(output_dir + '/list/' + name + '.pkl', 'wb') as f:
            pickle.dump(result, f)

    else:
        os.makedirs(output_dir + '/list/')
        with open(output_dir + '/list/' + name + '.pkl', 'wb') as f:
            pickle.dump(result, f)


def draw_plot(epoch_list, train_loss_list, train_acc_list, val_acc_list, test_loss_list, output_dir):
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(epoch_list, test_loss_list, label='test loss')
    plt.plot(epoch_list, train_loss_list, label='training loss')
    plt.legend()

    plt.subplot(122)
    plt.plot(epoch_list, train_acc_list, label='train acc')
    plt.plot(epoch_list, val_acc_list, label='validation acc')
    plt.legend()

    if os.path.isdir(output_dir + '/plot/'):
        plt.savefig(output_dir + '/plot/res.png')

    else:
        os.makedirs(output_dir + '/plot/')
        plt.savefig(output_dir + '/plot/res.png')
    plt.close()


def main():
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--T', type=int, default=16, help='time step of neuron, (default: 5)')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs, (default: 100)')
    parser.add_argument('--c', type=int, default=32,
                        help='channel count for each node, (example: 78, 109, 154), (default: 154)')
    parser.add_argument('--N', type=int, default=7, help="Number of graph node (default n=32)")
    parser.add_argument('--learning-rate', type=float, default=0.001, help='learning rate, (default: 1e-1)')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size, (default: 100)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='Momentum for SGD. Adam will not use momentum')
    parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)',
                        dest='weight_decay')
    parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--data-dir', default="E:\\datasets\\DVS128Gesture", type=str)
    parser.add_argument('--M', type=int, default=2, help="Steps of graph diffusion")
    parser.add_argument('--K', type=int, default=5, help="Top-K pruning")
    parser.add_argument('--trace', type=float, default=0.6, help="Trace-decay")
    parser.add_argument('--beta', type=float, default=0.2, help="beta_plasticity")

    # trace_decay = 0.6, beta_plasticity = 0.2
    args = parser.parse_args()

    config_and_results = {}

    device = args.device
    current_time = datetime.datetime.now().strftime("%m%d_%H%M")
    output_dir = "./DVSGesture/" + "(" + str(args.N + 2) + "," + ")" + "C" + str(
        args.c) + "T" + str(args.T) + "_time" + current_time

    os.makedirs(output_dir, exist_ok=True)


    import smodel

    net = smodel.Model(node_num=7, in_channels=2, out_channels=32, num_classes=11).to(device)
    size = 128
    train_set = DVS128Gesture(root=args.data_dir, train=True, data_type='frame', frames_number=args.T,
                              split_by='number')
    test_set = DVS128Gesture(root=args.data_dir, train=False, data_type='frame', frames_number=args.T,
                             split_by='number')

    train_sampler = torch.utils.data.RandomSampler(train_set)
    test_sampler = torch.utils.data.SequentialSampler(test_set)

    train_data_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=0, pin_memory=True)

    test_data_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=0, pin_memory=True)

    print(len(train_set))
    print(len(test_set))
    loss_fn = nn.CrossEntropyLoss().to(device)

    epochs = args.epochs
    # 5e-4
    optimizer = torch.optim.SGD(
        net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=64, gamma=0.1)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=64)

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

    x = torch.round(torch.rand((args.T, 1, 2, 128, 128))).to(device=args.device)
    Flops, params = profile(net, inputs=(x,))
    print('Flops: %.4fG' % (Flops / 1e9))
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
            img = img.to(device, dtype=torch.float32)
            img = img.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]

            label = label.to(device)
            label_onehot = F.one_hot(label, 11).float()
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
                frame = frame.to(device, dtype=torch.float32)
                frame = frame.transpose(0, 1)
                label = label.to(device)
                label_onehot = F.one_hot(label, 11).float()
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
                torch.save(net.state_dict(), output_dir + "/best_model_ablation.pth")

        draw_plot(epoch_list, train_loss_list, train_acc_list, test_acc_list, test_loss_list, output_dir)

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

    save_res_list(train_loss_list, 'trainLoss', output_dir)
    save_res_list(train_acc_list, 'trainAcc', output_dir)
    save_res_list(test_loss_list, 'testLoss', output_dir)
    save_res_list(test_acc_list, 'testAcc', output_dir)

    print("train_time:", time_str)
    with open(output_dir + '/config_and_results.txt', 'a') as file:
        for key, value in config_and_results.items():
            dict_str = json.dumps({key: value}, indent=4)
            file.write(dict_str + '\n')


if __name__ == "__main__":
    main()
