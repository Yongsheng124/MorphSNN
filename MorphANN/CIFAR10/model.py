import torch
import torch.nn as nn
import torch.nn.functional as F


# from spikingjelly.clock_driven import layer # Comment out this line if SNN components are not used, or keep it
# from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode # Same as above

# Simple ANN component simulation (if you don't have spikingjelly environment, use this placeholder, otherwise keep above imports)
# If you have spikingjelly, uncomment the line below and comment out the class layer
# from spikingjelly.clock_driven import layer

def conv3x3_snn(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.GELU(),
    )


class Pooling_Node(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Pooling_Node, self).__init__()
        self.conv = conv3x3_snn(in_channels, out_channels)
        self.pool = nn.AvgPool2d(2, 2)

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

        # 1. Define nodes
        self.module_list = nn.ModuleList([Pooling_Node(self.in_channels, self.out_channels)])
        for _ in range(self.total_nodes - 1):
            self.module_list.append(conv3x3_snn(self.out_channels, self.out_channels))

        # 2. [Fix] Move parameter definition out of loop, and initialize with correct shape
        self.output_weights = nn.Parameter(torch.ones(self.total_nodes, requires_grad=True))

    def compute_diffusion_operator(self, adj):
        B, N, _ = adj.shape
        # 1. Force symmetry
        adj_sym = (adj + adj.transpose(1, 2)) / 2
        # I = torch.eye(N, device=adj.device).unsqueeze(0).expand(B, N, N) # Self-loop optional
        adj_hat = adj_sym
        degree = adj_hat.sum(dim=2).clamp(min=1e-6)  # Prevent division by 0
        degree_inv_sqrt = torch.pow(degree, -0.5)
        D_inv_sqrt_mat = torch.diag_embed(degree_inv_sqrt)
        diffusion_op = torch.bmm(torch.bmm(D_inv_sqrt_mat, adj_hat), D_inv_sqrt_mat)
        return diffusion_op

    def forward(self, x):
        B, C, H, W = x.shape
        N = self.total_nodes

        # Initialize uniform graph
        S = torch.ones(B, N, N, device=x.device) / N

        # Node 0 (downsampling node)
        out_0_t = self.module_list[0](x)  # [B, OutC, H/2, W/2]

        # [Fix] Safer flattening method, avoid potential risks of squeeze(0)
        out_flat = out_0_t.flatten(1)  # [B, Feature_Dim]

        # Prepare diffusion source
        X_source = torch.zeros(B, N, out_flat.shape[1], device=x.device)
        X_source[:, 0, :] = out_flat

        # Diffusion
        diffusion_op = self.compute_diffusion_operator(S)
        H_diffused = torch.bmm(diffusion_op, X_source)
        H_diffused = torch.bmm(diffusion_op, H_diffused)  # 2-step diffusion
        mat_t = H_diffused

        out_list_t = [out_0_t]

        # [Fix] Correct loop indentation
        for i in range(1, N):
            # Restore feature map shape from diffusion matrix
            in_i_t = mat_t[:, i, :].reshape(B, self.out_channels, out_0_t.shape[2], out_0_t.shape[3])
            out_i_t = self.module_list[i](in_i_t)
            out_list_t.append(out_i_t)

        # Weighted output
        total_out = 0.
        # Use Sigmoid to normalize weights
        weights = torch.sigmoid(self.output_weights)
        for i in range(N):
            total_out += out_list_t[i] * weights[i]

        return total_out


class ResidualWrapper(nn.Module):
    def __init__(self, node_num, in_channels, out_channels):
        super().__init__()
        self.body = STSPBlock(node_num, in_channels, out_channels)
        self.shortcut = nn.Sequential()
        # STSPBlock always downsamples (Pool Node 0), so Shortcut must match
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(2, 2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.AvgPool2d(2, 2)

    def forward(self, x):
        return F.relu(self.body(x) + self.shortcut(x))


class BetterModel(nn.Module):
    def __init__(self, node_num=7, in_channels=3, base_channels=64, num_classes=10):
        super(BetterModel, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU()
        )

        self.stage1 = ResidualWrapper(node_num, base_channels, base_channels * 2)
        self.stage2 = ResidualWrapper(node_num, base_channels * 2, base_channels * 4)
        self.stage3 = ResidualWrapper(node_num, base_channels * 4, base_channels * 8)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)

        final_dim = base_channels * 8

        # self.fc = nn.Sequential(
        #     nn.Linear(final_dim, final_dim),
        #     nn.BatchNorm1d(final_dim),
        #     nn.GELU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(final_dim, num_classes)
        # )

        self.fc = nn.Sequential(
            nn.Linear(final_dim, 10),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    net = BetterModel(node_num=4, base_channels=64).to(device)

    # Test Batch Size = 2
    x = torch.randn(2, 3, 32, 32).to(device)
    y = net(x)
    print(f"Output shape: {y.shape}")  # [2, 10]

    # Simple parameter validation
    from torchinfo import summary

    summary(net, input_size=(2, 3, 32, 32))
    from thop import profile
    # FLOPs and Params check
    x_dummy = torch.randn(1, 3, 32, 32).to(device)
    Flops, params = profile(net, inputs=(x_dummy,))
    print('------------------------------------------------')
    print('FLOPs: %.4fG' % (Flops / 1e9))
    print('Params: %.4fM' % (params / 1e6))
    print('------------------------------------------------')