import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import layer
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode


def conv3x3_snn(in_channels, out_channels):
    """标准的 Conv-BN-LIF 模块"""
    return nn.Sequential(
        layer.SeqToANNContainer(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ),
        MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True)
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

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = self.total_nodes
        S = torch.ones(B, N, N, device=x.device) / N
        final_outputs = []
        for t in range(T):
            x_t = x[t:t + 1]

            out_0_t = self.module_list[0](x_t)

            out_flat = out_0_t.squeeze(0).view(B, -1)  # [B, Feature_Dim]
            X_source = torch.zeros(B, N, out_flat.shape[1], device=x.device)
            X_source[:, 0, :] = out_flat  # 只有节点0有输入，作为"热源"
            H_diffused = torch.bmm(S, X_source)
            H_diffused = torch.bmm(S, H_diffused)
            mat_t = H_diffused
            out_list_t = [out_0_t]
            for i in range(1, N):
                in_i_t = mat_t[:, i, :].reshape(B, self.out_channels, out_0_t.shape[-2], out_0_t.shape[-1]).unsqueeze(0)
                out_i_t = self.module_list[i](in_i_t)
                out_list_t.append(out_i_t)
            # 7个节点的输出
            total_out = 0.
            for i in range(N):
                total_out += out_list_t[i] * torch.sigmoid(self.output_weights[i])
            final_outputs.append(total_out)
        output = torch.cat(final_outputs, dim=0)
        return output


class Model(nn.Module):
    def __init__(self, node_num=5, in_channels=2, out_channels=64, num_classes=11):
        super(Model, self).__init__()

        self.conv1 = nn.Sequential(
            layer.SeqToANNContainer(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
            ),
            MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True)
        )
        self.pool1 = layer.SeqToANNContainer(nn.AvgPool2d(2, 2))

        self.stsp_block = STSPBlock(node_num, out_channels, out_channels)

        self.stsp_block2 = STSPBlock(node_num, out_channels, out_channels)

        self.stsp_block3 = STSPBlock(node_num, out_channels, out_channels)

        self.pool2 = layer.SeqToANNContainer(nn.AvgPool2d(2, 2))

        self.flatten = nn.Flatten(2)
        self.fc = nn.Linear(out_channels, num_classes, bias=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)  # Size / 2

        out = self.stsp_block(out)
        out = self.pool2(out)

        out = self.stsp_block2(out)
        out = self.pool2(out)

        out = self.stsp_block3(out)
        out = self.pool2(out)
        out = self.flatten(out)
        out = out.mean(0)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    # 简单的测试桩
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # DVS Gesture 输入模拟: [T=16, B=2, C=2, H=32, W=32]
    x = torch.rand((5, 1, 2, 128, 128)).to(device)
    x = (x > 0.8).float()  # 模拟稀疏脉冲

    net = Model(node_num=7, in_channels=2, out_channels=16, num_classes=11).to(device)

    # 这里的 8*8 需要根据上面的 input 32 调整，32/2/2 = 8，所以代码里的 8*8 是对的

    y = net(x)
    print(y.shape)
