import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import layer
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode


# ==========================================
# 1. 基础组件
# ==========================================
class SymmetricGATbuilder(nn.Module):
    """
    用于构建动态邻接矩阵的对称图注意力层。
    区别于标准 GAT：
    1. 输出的是 Adjacency Matrix (N x N) 而不是 Feature。
    2. 强制对称性，以支持无向图拉普拉斯扩散。
    """

    def __init__(self, in_features, num_heads=4, dropout=0.2):
        super(SymmetricGATbuilder, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_features // num_heads

        assert self.head_dim * num_heads == in_features, "in_features must be divisible by num_heads"

        # 线性变换 W
        self.W = nn.Linear(in_features, in_features, bias=False)

        # 注意力向量 a (用于 Additive Attention)
        # 我们使用 [2 * head_dim] -> [1] 的映射
        self.a = nn.Parameter(torch.zeros(num_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.a)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        # h: [B, N, C] (这里是 Trace)
        B, N, C = h.shape

        # 1. 线性变换 & 分头
        # [B, N, C] -> [B, N, Heads, Head_Dim]
        h_prime = self.W(h).view(B, N, self.num_heads, self.head_dim)

        # 2. 构建全连接对 (All-pairs)
        # 为了高效计算 Additive Attention: a^T [Wh_i || Wh_j]
        # 等价于: a1^T Wh_i + a2^T Wh_j

        # 将参数 a 分为 a_1 和 a_2
        a_1 = self.a[:, :self.head_dim]  # [Heads, Head_Dim]
        a_2 = self.a[:, self.head_dim:]  # [Heads, Head_Dim]

        # 计算 a_1 * Wh_i -> [B, N, Heads]
        # einsum: 'bnhd, hd -> bnh'
        e_1 = torch.einsum('bnhd,hd->bnh', h_prime, a_1)
        e_2 = torch.einsum('bnhd,hd->bnh', h_prime, a_2)

        # 广播加法得到 N x N 矩阵
        # [B, N, 1, Heads] + [B, 1, N, Heads] -> [B, N, N, Heads]
        e = e_1.unsqueeze(2) + e_2.unsqueeze(1)
        e = self.leakyrelu(e)

        # 3. 【核心步骤】强制对称 (Symmetrization)
        # GAT 原生是不对称的，这里我们取平均：(e_ij + e_ji) / 2
        # 这在物理上意味着：i 对 j 的关注 和 j 对 i 的关注 达成共识
        e_sym = (e + e.permute(0, 2, 1, 3)) / 2

        # 4. 归一化 (Softmax over neighbors)
        # [B, N, N, Heads] -> Mean over Heads -> [B, N, N]
        # 也可以先 Softmax 再 Mean，这里先 Mean 节省计算
        attention = e_sym.mean(dim=-1)

        # 5. Temperature Scaling (可选，让图更稀疏)
        attention = F.softmax(attention / 0.5, dim=-1)

        return self.dropout(attention)


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
    def __init__(self, node_num, in_channels, out_channels, k_pruning=3, beta_plasticity=0.0, trace_decay=0.6):
        super(STSPBlock, self).__init__()
        self.node_num = node_num
        self.total_nodes = self.node_num + 2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k_pruning
        self.attn_norm = nn.LayerNorm(out_channels)
        self.gat_builder = SymmetricGATbuilder(out_channels, num_heads=4)
        # === [核心参数] ===
        self.trace_decay = trace_decay
        self.beta_param = nn.Parameter(torch.tensor(beta_plasticity), requires_grad=False)

        # === [特征映射器] (保持不变) ===
        self.spatial_pool = layer.SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.feature_transform = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU()
        )

        # === [节点定义] (保持不变) ===
        self.module_list = nn.ModuleList([Pooling_Node(self.in_channels, self.out_channels)])
        for _ in range(self.total_nodes - 1):
            self.module_list.append(conv3x3_snn(self.out_channels, self.out_channels))

        self.output_weights = nn.Parameter(torch.ones(self.total_nodes, requires_grad=True))

        # ============================================================
        # 【新增修改 1】: 扩散控制参数
        # ============================================================
        # gamma: 控制有多少信息来自于扩散，有多少保留原始输入
        # 初始化为 0.5 表示平衡
        self.diffusion_gamma = nn.Parameter(torch.tensor(0.5))

    # ============================================================
    # 【新增修改 2】: 对称归一化拉普拉斯算子计算工具
    # ============================================================
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
        adj_hat = adj_sym + I

        # 3. 计算度矩阵 D (Degree Matrix)
        degree = adj_hat.sum(dim=2)  # [B, N]

        # 4. 计算 D^(-1/2)
        degree_inv_sqrt = torch.pow(degree + 1e-6, -0.5)  # 加 epsilon 防止除0
        D_inv_sqrt_mat = torch.diag_embed(degree_inv_sqrt)  # [B, N, N]

        # 5. 对称归一化: D^(-1/2) * A_hat * D^(-1/2)
        diffusion_op = torch.bmm(torch.bmm(D_inv_sqrt_mat, adj_hat), D_inv_sqrt_mat)

        return diffusion_op
    
    def get_graph_entropy_loss(self, graph_tensor):
        """
        计算图权重的熵，作为正则项。
        目标：最小化熵 -> 迫使图变得稀疏、确信 (Confident)。
        graph_tensor: [B, T, N, N] (Attention Scores)
        """
        # 加上 epsilon 防止 log(0)
        eps = 1e-8
        # H = - sum( p * log(p) )
        # 我们希望 p 接近 0 或 1，这样 H 最小
        entropy = -torch.sum(graph_tensor * torch.log(graph_tensor + eps), dim=-1)
        return entropy.mean()

    def prune_topk(self, S, k):
        # (保持不变)
        top_k_vals, _ = torch.topk(S, k, dim=-1)
        k_th_val = top_k_vals[:, -1].unsqueeze(1)
        mask = (S >= k_th_val).float()
        return S * mask

    def compute_trace_attention(self, traces):
        # (保持不变)
        traces = self.attn_norm(traces)
        d_k = traces.size(-1)
        scores = torch.bmm(traces, traces.transpose(1, 2))
        temperature = 0.05
        scores = scores / ((d_k ** 0.5) * temperature)
        return F.softmax(scores, dim=-1)

    def forward(self, x, return_graph=False):
        T, B, C, H, W = x.shape
        N = self.total_nodes

        S = torch.ones(B, N, N, device=x.device) / N
        node_traces = torch.zeros(B, N, self.out_channels, device=x.device)
        beta = self.beta_param

        final_outputs = []
        graph_snapshots = []

        for t in range(T):
            x_t = x[t:t + 1]

            # Phase 1: Update Input Node & Trace (保持不变)
            out_0_t = self.module_list[0](x_t)
            feat_0_raw = self.spatial_pool(out_0_t).squeeze(0)
            feat_0_vec = self.feature_transform(feat_0_raw)
            node_traces[:, 0, :] = self.trace_decay * node_traces[:, 0, :] + \
                                   (1 - self.trace_decay) * feat_0_vec

            # Phase 2: Structural Plasticity (保持不变)
            #alpha_t = self.compute_trace_attention(node_traces)
            alpha_t = self.gat_builder(node_traces)
            S = beta * S.detach() + (1 - beta) * alpha_t

            if return_graph:
                graph_snapshots.append(S.detach())

            S_pruned = torch.stack([self.prune_topk(S[b], self.k) for b in range(B)])

            # ============================================================
            # Phase 3: 【核心修改】基于拉普拉斯扩散的图传播
            # ============================================================

            # 1. 准备源信号 (Source Signal)
            out_flat = out_0_t.squeeze(0).view(B, -1)  # [B, Feature_Dim]
            X_source = torch.zeros(B, N, out_flat.shape[1], device=x.device)
            X_source[:, 0, :] = out_flat  # 只有节点0有输入，作为"热源"

            # 2. 计算扩散算子 (Diffusion Operator)
            # 这里调用新写的函数，将有向的 S_pruned 转化为 无向归一化的 P
            # 这一步对应理论中的: 最小化 Dirichlet Energy

           
            diffusion_op = self.compute_diffusion_operator(S_pruned)
            
            
            
            # print(diffusion_op)
            #  将对应一个很漂亮的理论。
            H_diffused = torch.bmm(diffusion_op, X_source)
            H_diffused = torch.bmm(diffusion_op, H_diffused)


            # 4. 源注入与扩散融合 (Source Injection & Fusion)
            # 公式: H_out = (1 - gamma) * X_source + gamma * H_diffused
            # 物理含义: 保证源节点信息不丢失(Retain)，同时让信息扩散到全网(Diffuse)
            gamma = torch.sigmoid(self.diffusion_gamma)
            # mat_t = (1 - gamma) * X_source + gamma * H_diffused
            mat_t = H_diffused
            # ============================================================
            # Phase 3 结束
            # ============================================================

            out_list_t = [out_0_t]
            feats_instant_list = [feat_0_vec]

            for i in range(1, N):
                # Reshape 回 [B, C, H, W] (保持不变)
                in_i_t = mat_t[:, i, :].reshape(B, self.out_channels, out_0_t.shape[-2], out_0_t.shape[-1]).unsqueeze(0)

                out_i_t = self.module_list[i](in_i_t)
                out_list_t.append(out_i_t)

                feat_i_raw = self.spatial_pool(out_i_t).squeeze(0)
                feat_i_vec = self.feature_transform(feat_i_raw)
                feats_instant_list.append(feat_i_vec)

            # Phase 4: Update Traces & Aggregate (保持不变)
            feats_new = torch.stack(feats_instant_list, dim=1)
            node_traces = self.trace_decay * node_traces + (1 - self.trace_decay) * feats_new

            stacked_outputs = torch.stack(out_list_t).squeeze(1)
            unflattened = stacked_outputs.permute(1, 0, 2, 3, 4)
            weights = torch.sigmoid(self.output_weights).view(1, N, 1, 1, 1)
            y_t = (unflattened * weights).sum(dim=1)

            final_outputs.append(y_t.unsqueeze(0))

        output = torch.cat(final_outputs, dim=0)

        if return_graph:
            return output, torch.stack(graph_snapshots).permute(1, 0, 2, 3)

        return output


# ==========================================
# 3. 整体模型 (DVS Gesture 适配版)
# ==========================================

class Model(nn.Module):
    def __init__(self, node_num=5, in_channels=2, out_channels=64, num_classes=11):
        super(Model, self).__init__()

        # Stem Layer: 处理 DVS 原始输入 (2通道: On/Off Events)
        self.conv1 = nn.Sequential(
            layer.SeqToANNContainer(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
            ),
            MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True)
        )
        self.pool1 = layer.SeqToANNContainer(nn.AvgPool2d(2, 2))

        # 核心动态图模块
        # 注意: beta_plasticity=0.0 让 Trace 主导
        self.stsp_block = STSPBlock(node_num, out_channels, out_channels,
                                    k_pruning=3, beta_plasticity=0.9, trace_decay=0.6)

        self.stsp_block2 = STSPBlock(node_num, out_channels, out_channels,
                                    k_pruning=3, beta_plasticity=0.9, trace_decay=0.6)

        self.stsp_block3 = STSPBlock(node_num, out_channels, out_channels,
                                     k_pruning=3, beta_plasticity=0.9, trace_decay=0.6)


        self.pool2 = layer.SeqToANNContainer(nn.AvgPool2d(2, 2))

    
        self.flatten = nn.Flatten(2)
        self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, x, return_graph=False):
        # x: [T, B, 2, 128, 128] (假设)

        out = self.conv1(x)
        out = self.pool1(out)  # Size / 2

        if return_graph:
            out, graph_tensor = self.stsp_block(out, return_graph=True)
        else:
            out = self.stsp_block(out)

        out = self.pool2(out)
        out = self.stsp_block2(out)
        out = self.pool2(out)
        out = self.stsp_block3(out)
        out = self.pool2(out)
        out = self.flatten(out)
        out_mean = out.mean(0)
      

        return self.fc(out_mean) if not return_graph else (self.fc(out_mean), graph_tensor)


if __name__ == '__main__':
    # 简单的测试桩
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # DVS Gesture 输入模拟: [T=16, B=2, C=2, H=32, W=32]
    x = torch.rand((16, 2, 2, 128, 128)).to(device)
    x = (x > 0.8).float()  # 模拟稀疏脉冲

    net = Model(node_num=5, in_channels=2, out_channels=32, num_classes=11).to(device)

    # 这里的 8*8 需要根据上面的 input 32 调整，32/2/2 = 8，所以代码里的 8*8 是对的

    y, graph = net(x, return_graph=True)
    print(f"Output Shape: {y.shape}")
    print(f"Graph Shape: {graph.shape} (Batch, Time, Node, Node)")

    # 检查一下 Graph 是否全为 0 (DVS 常见坑)
    print(f"Graph Mean: {graph.mean().item()}")
    print(f"Graph Std: {graph.std().item()}")