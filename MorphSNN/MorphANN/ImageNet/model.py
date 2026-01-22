import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint  # 显存救星


# ==========================================
# 0. 基础组件
# ==========================================
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def conv3x3_bn(in_channels, out_channels, stride=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False, groups=groups),
        nn.BatchNorm2d(out_channels),
        nn.GELU(),
    )


class Pooling_Node(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Pooling_Node, self).__init__()
        self.conv = conv3x3_bn(in_channels, out_channels, groups=groups)
        if stride > 1:
            self.pool = nn.AvgPool2d(3, 2, 1)
        else:
            self.pool = nn.Identity()

    def forward(self, x):
        return self.pool(self.conv(x))


class Node(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(Node, self).__init__()
        self.conv = conv3x3_bn(in_channels, out_channels, groups=groups)

    def forward(self, x):
        return self.conv(x)


# ==========================================
# 1. 你的核心 STSP Block (包含完整的图扩散)
# ==========================================
class STSPBlock_ImageNet(nn.Module):
    def __init__(self, node_num, in_channels, out_channels, stride=1, groups=1):
        super(STSPBlock_ImageNet, self).__init__()
        self.node_num = node_num
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # Node 0: 入口节点
        self.module_list = nn.ModuleList([
            Pooling_Node(in_channels, out_channels, stride=stride, groups=max(1, groups // 2))
        ])

        # Node 1~N: 并行节点
        for _ in range(node_num - 1):
            self.module_list.append(conv3x3_bn(out_channels, out_channels, groups=groups))

        self.output_weights = nn.Parameter(torch.ones(node_num, requires_grad=True))
        self.se = SEBlock(out_channels)

    def compute_diffusion_operator(self, adj):
        B, N, _ = adj.shape
        adj_sym = (adj + adj.transpose(1, 2)) / 2
        adj_hat = adj_sym
        degree = adj_hat.sum(dim=2).clamp(min=1e-6)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        D_inv_sqrt_mat = torch.diag_embed(degree_inv_sqrt)
        diffusion_op = torch.bmm(torch.bmm(D_inv_sqrt_mat, adj_hat), D_inv_sqrt_mat)
        return diffusion_op

    def forward(self, x):
        # 如果显存不够，可以用 checkpoint 包装整个 forward
        # return checkpoint(self._forward_impl, x)
        return self._forward_impl(x)

    def _forward_impl(self, x):
        B, _, _, _ = x.shape
        N = self.node_num

        # 1. 初始化图结构 (全连接均匀初始化)
        # 这里你可以改成可学习的 parameter，如果你想做结构可塑性
        S = torch.ones(B, N, N, device=x.device) / N

        # 2. Node 0 处理
        out_0 = self.module_list[0](x)  # [B, C_out, H, W]
        B, C, H, W = out_0.shape

        # 3. 准备扩散源
        # 关键优化：为了 ImageNet 速度，我们不要 flatten 整个 feature map
        # 我们对 spatial 维度保持独立，或者只在 Channel 维度扩散
        # 但既然你要求"你的逻辑"，我这里严格按照你之前的逻辑写：Flatten (H,W)
        out_flat = out_0.flatten(1)  # [B, C*H*W]

        X_source = torch.zeros(B, N, out_flat.shape[1], device=x.device)  # [B, N, Features]
        X_source[:, 0, :] = out_flat

        # 4. 计算扩散算子
        diffusion_op = self.compute_diffusion_operator(S)  # [B, N, N]

        # 5. 执行图扩散 (矩阵乘法)
        # [B, N, N] x [B, N, Feat] -> [B, N, Feat]
        H_diffused = torch.bmm(diffusion_op, X_source)
        H_diffused = torch.bmm(diffusion_op, H_diffused)  # 2步扩散

        mat_t = H_diffused  # [B, N, C*H*W]

        out_list = [out_0]

        # 6. 分发给其他节点并计算
        for i in range(1, N):
            # 恢复形状 [B, C, H, W]
            in_i = mat_t[:, i, :].view(B, C, H, W)
            out_i = self.module_list[i](in_i)
            out_list.append(out_i)

        # 7. 加权聚合
        total_out = 0.
        weights = torch.sigmoid(self.output_weights)
        for i in range(N):
            total_out += out_list[i] * weights[i]

        return self.se(total_out)


class ResidualWrapper(nn.Module):
    def __init__(self, node_num, in_channels, out_channels, stride=1, groups=1):
        super().__init__()
        self.body = STSPBlock_ImageNet(node_num, in_channels, out_channels, stride, groups)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(stride, stride) if stride > 1 else nn.Identity(),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return F.relu(self.body(x) + self.shortcut(x))


# ==========================================
# 2. ImageNet 主模型
# ==========================================
class BetterModel_ImageNet(nn.Module):
    def __init__(self, node_num=4, num_classes=1000):
        super(BetterModel_ImageNet, self).__init__()

        base_channels = 64

        # Deep Stem (3层卷积) - 标配
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Stacking Stages (ResNet-50 风格堆叠)
        # 你的 STSPBlock 比较重（有矩阵乘法），所以堆叠层数比 ResNet 少一点，
        # 但通道数保持足够，利用图结构弥补深度的不足。

        # Stage 1: 56x56
        self.stage1 = self._make_layer(base_channels, 128, 2, stride=1, node_num=node_num, groups=4)

        # Stage 2: 28x28
        self.stage2 = self._make_layer(128, 256, 3, stride=2, node_num=node_num, groups=8)

        # Stage 3: 14x14
        self.stage3 = self._make_layer(256, 512, 4, stride=2, node_num=node_num, groups=16)

        # Stage 4: 7x7
        self.stage4 = self._make_layer(512, 1024, 2, stride=2, node_num=node_num, groups=32)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(1024, num_classes)

        self._init_weights()

    def _make_layer(self, in_c, out_c, blocks, stride, node_num, groups):
        layers = []
        # 第一个 Block 处理 stride 和通道变化
        layers.append(ResidualWrapper(node_num, in_c, out_c, stride=stride, groups=groups))
        # 后续堆叠
        for _ in range(1, blocks):
            layers.append(ResidualWrapper(node_num, out_c, out_c, stride=1, groups=groups))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # ImageNet 测试
    model = BetterModel_ImageNet(node_num=4, num_classes=1000).to(device)

    # 模拟输入 (Batch=2, 3, 224, 224)
    x = torch.randn(2, 3, 224, 224).to(device)
    y = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")

    from thop import profile

    x_dummy = torch.randn(1, 3, 224, 224).to(device)
    flops, params = profile(model, inputs=(x_dummy,))
    print('------------------------------------------------')
    print(f'FLOPs:  {flops / 1e9:.2f} G')
    print(f'Params: {params / 1e6:.2f} M')
    print('------------------------------------------------')