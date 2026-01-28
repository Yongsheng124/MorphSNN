import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import layer
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode
from thop import profile


class SEBlock_T(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock_T, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Input x: [T, B, C, H, W]
        T, B, C, H, W = x.shape

        # 1. Take mean along T dimension (Mean Firing Rate) -> [B, C, H, W]
        x_mean = x.mean(0)

        # 2. Squeeze: Global Avg Pooling -> [B, C]
        y = self.avg_pool(x_mean).view(B, C)

        # 3. Excitation: FC layers -> [B, C, 1, 1]
        y = self.fc(y).view(B, C, 1, 1)

        # 4. Scale: Broadcast weights back to T dimension -> [T, B, C, H, W]
        # unsqueeze(0) adds T dimension
        return x * y.unsqueeze(0).expand_as(x)


def conv3x3_snn(in_channels, out_channels):
    """Standard Conv-BN-LIF module"""
    return nn.Sequential(
        layer.SeqToANNContainer(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ),
        MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend="torch")
    )


class Pooling_Node(nn.Module):
    """Pooling node for size reduction"""

    def __init__(self, in_channels, out_channels):
        super(Pooling_Node, self).__init__()
        self.conv = conv3x3_snn(in_channels, out_channels)
        self.pool = layer.SeqToANNContainer(nn.AvgPool2d(2, 2))

    def forward(self, x):
        return self.pool(self.conv(x))


# ==========================================
# 2. Core: Spatio-Temporal Structural Plasticity Block (STSP)
# ==========================================

class STSPBlock(nn.Module):
    def __init__(self, node_num, in_channels, out_channels):
        super(STSPBlock, self).__init__()
        self.node_num = node_num
        self.total_nodes = self.node_num
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.se_block = SEBlock_T(out_channels)
        # === [Node Definition] (Unchanged) ===
        self.module_list = nn.ModuleList([Pooling_Node(self.in_channels, self.out_channels)])
        for _ in range(self.total_nodes - 1):
            self.module_list.append(conv3x3_snn(self.out_channels, self.out_channels))

        self.output_weights = nn.Parameter(torch.ones(self.total_nodes, requires_grad=True))

    def compute_diffusion_operator(self, adj):
        B, N, _ = adj.shape

        # 1. Force symmetry (Symmetrization): Make the graph undirected, ensuring energy function existence
        adj_sym = (adj + adj.transpose(1, 2)) / 2

        # 2. Add self-loops: Similar to GCN, enhance preservation of node's own features, prevent division by 0
        I = torch.eye(N, device=adj.device).unsqueeze(0).expand(B, N, N)
        # adj_hat = adj_sym + I
        adj_hat = adj_sym
        # 3. Calculate degree matrix D (Degree Matrix)
        degree = adj_hat.sum(dim=2)  # [B, N]

        # 4. Calculate D^(-1/2)
        degree_inv_sqrt = torch.pow(degree + 1e-6, -0.5)  # Add epsilon to prevent division by 0
        D_inv_sqrt_mat = torch.diag_embed(degree_inv_sqrt)  # [B, N, N]

        # 5. Symmetric normalization: D^(-1/2) * A_hat * D^(-1/2)
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
        X_source[:, 0, :] = out_flat  # Only node 0 has input, acting as "heat source"
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

        return self.se_block(total_out), out_list_t


class Model(nn.Module):
    def __init__(self, node_num=5, in_channels=2, out_channels=64, num_classes=11):
        super(Model, self).__init__()

        self.conv1 = nn.Sequential(
            layer.SeqToANNContainer(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
            ),
            MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend="torch")
        )
        self.pool1 = layer.SeqToANNContainer(nn.AvgPool2d(2, 2))

        self.stage1 = STSPBlock(node_num, out_channels, out_channels * 2)

        self.stage2 = STSPBlock(node_num, out_channels * 2, out_channels * 4)

        self.stage3 = STSPBlock(node_num, out_channels * 4, out_channels * 8)

        self.pool2 = layer.SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))

        self.flatten = nn.Flatten(2)
        final_dim = out_channels * 8
        self.fc = nn.Sequential(
            nn.Linear(final_dim, 10),
        )

    def forward(self, x):
        # x: [T, B, 2, 128, 128] (assumed)
        x = (x.unsqueeze(0)).repeat(4, 1, 1, 1, 1)

        out = self.conv1(x)

        out, vs1 = self.stage1(out)

        out, vs2 = self.stage2(out)

        out, vs3 = self.stage3(out)

        out = self.pool2(out)
        out = self.flatten(out)
        out_mean = out.mean(0)
        out = self.fc(out_mean)
        return out, vs1


if __name__ == '__main__':
    # Simple test stub
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    net = Model(node_num=4, in_channels=3, out_channels=64, num_classes=10).to(device)

    x = torch.round(torch.rand((2, 3, 32, 32))).to(device)
    Flops, params = profile(net, inputs=(x,))
    print('Flops: %.4fG' % (Flops / 1e9))  # Convert FLOPs to GFLOPs
    print('params: %.4fM' % (params / 1000000))