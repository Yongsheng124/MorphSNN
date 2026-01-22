import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import layer
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode
from thop import profile


def conv3x3_snn(in_channels, out_channels):
    """标准的 Conv-BN-LIF 模块"""
    return nn.Sequential(
        layer.SeqToANNContainer(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ),
        MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend="torch")
    )


class Pooling_Node(nn.Module):
    """带池化的节点，用于缩减尺寸"""

    def __init__(self, in_channels, out_channels):
        super(Pooling_Node, self).__init__()
        self.conv = conv3x3_snn(in_channels, out_channels)
        self.pool = layer.SeqToANNContainer(nn.AvgPool2d(2, 2))

    def forward(self, x):
        return self.pool(self.conv(x))


# ==========================================
# 2. 核心：Spatio-Temporal Structural Plasticity Block (STSP)
# ==========================================

class STSPBlock(nn.Module):
    def __init__(self, node_num, in_channels, out_channels):
        super(STSPBlock, self).__init__()
        self.node_num = node_num
        self.total_nodes = self.node_num
        self.in_channels = in_channels
        self.out_channels = out_channels

        # === [节点定义] (保持不变) ===
        self.module_list = nn.ModuleList([Pooling_Node(self.in_channels, self.out_channels)])
        for _ in range(self.total_nodes - 1):
            self.module_list.append(conv3x3_snn(self.out_channels, self.out_channels))

        self.output_weights = nn.Parameter(torch.ones(self.total_nodes, requires_grad=True))

    def compute_diffusion_operator(self, adj):
        """
        将任意的邻接矩阵转化为对称归一化的扩散算子。
        公式: P = D^(-1/2) * (A_sym + I) * D^(-1/2)
        这保证了图是无向的，且扩散过程数值稳定 (谱半径 <= 1)。
        """
        B, N, _ = adj.shape

        # 1. 强制对称 (Symmetrization): 使得图变为无向图，保证能量函数存在
        adj_sym = (adj + adj.transpose(1, 2)) / 2

        # 2. 添加自环 (Self-loop): 类似于 GCN，增强节点自身特征的保留，防止除以0
        I = torch.eye(N, device=adj.device).unsqueeze(0).expand(B, N, N)
        # adj_hat = adj_sym + I
        adj_hat = adj_sym
        # 3. 计算度矩阵 D (Degree Matrix)
        degree = adj_hat.sum(dim=2)  # [B, N]

        # 4. 计算 D^(-1/2)
        degree_inv_sqrt = torch.pow(degree + 1e-6, -0.5)  # 加 epsilon 防止除0
        D_inv_sqrt_mat = torch.diag_embed(degree_inv_sqrt)  # [B, N, N]

        # 5. 对称归一化: D^(-1/2) * A_hat * D^(-1/2)
        diffusion_op = torch.bmm(torch.bmm(D_inv_sqrt_mat, adj_hat), D_inv_sqrt_mat)

        return diffusion_op

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = self.total_nodes
        S = torch.ones(B, N, N, device=x.device) / N
        final_outputs = []
        out_0_t = self.module_list[0](x)
        out_flat = out_0_t.squeeze(0).view(B, -1)  # [B, Feature_Dim]
        X_source = torch.zeros(B, N, out_flat.shape[1], device=x.device)
        X_source[:, 0, :] = out_flat  # 只有节点0有输入，作为"热源"
        # S = (S + S.transpose(-2, -1)) / 2
        diffusion_op = self.compute_diffusion_operator(S)
        #
        H_diffused = torch.bmm(diffusion_op, X_source)
        H_diffused = torch.bmm(diffusion_op, H_diffused)
        mat_t = H_diffused
        out_list_t = [out_0_t]
        for i in range(1, N):
            in_i_t = mat_t[:, i, :].reshape(T, B, self.out_channels, out_0_t.shape[-2], out_0_t.shape[-1])
            out_i_t = self.module_list[i](in_i_t)
            out_list_t.append(out_i_t)
        total_out = 0.
        for i in range(N):
            total_out += out_list_t[i] * torch.sigmoid(self.output_weights[i])
        return total_out


class Model(nn.Module):
    def __init__(self, node_num=5, in_channels=2, out_channels=64, num_classes=1000):
        super(Model, self).__init__()

        self.conv1 = nn.Sequential(
            layer.SeqToANNContainer(
                nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3, stride=2, bias=False),
                nn.BatchNorm2d(out_channels),
            ),
            MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend="torch")
        )
        self.pool1 = layer.SeqToANNContainer(nn.AvgPool2d(2, 2))

        self.stage1 = STSPBlock(4, out_channels, out_channels)

        self.stage2 = STSPBlock(4, out_channels, out_channels * 2)

        self.stage3 = STSPBlock(4, out_channels * 2, out_channels * 4)

        self.stage4 = STSPBlock(4, out_channels * 4, out_channels * 8)

        self.pool2 = layer.SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))

        self.flatten = nn.Flatten(2)
        final_dim = out_channels * 8
        # self.fc = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.3),  # 稍微给点 Dropout
        #     nn.Linear(hidden_dim, num_classes)
        # )

        self.fc = nn.Sequential(
            nn.Linear(final_dim, final_dim),
            nn.BatchNorm1d(final_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(final_dim, num_classes)
        )

    def forward(self, x):
        # x: [T, B, 2, 128, 128] (假设)
        x = (x.unsqueeze(0)).repeat(4, 1, 1, 1, 1)

        out = self.conv1(x)



        out = self.stage1(out)

        out = self.stage2(out)

        out = self.stage3(out)

        out = self.stage4(out)
        out = self.pool2(out)
        out = self.flatten(out)
        out_mean = out.mean(0)
        out = self.fc(out_mean)
        return out


if __name__ == '__main__':
    # 简单的测试桩
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    net = Model(node_num=4, in_channels=3, out_channels=64, num_classes=1000).to(device)

    x = torch.round(torch.rand((2, 3, 64, 64))).to(device)
    Flops, params = profile(net, inputs=(x,))
    print('Flops: %.4fG' % (Flops / 1e9))  # 将 FLOPs 转换为 GFLOPs
    print('params参数量: % .4fM' % (params / 1000000))