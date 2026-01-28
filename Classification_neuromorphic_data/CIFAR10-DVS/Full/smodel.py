import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import layer
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode


# ==========================================
# 1. Basic Components
# ==========================================
class SymmetricGATbuilder(nn.Module):
    """
    Symmetric Graph Attention Layer for building dynamic adjacency matrix.
    Differences from standard GAT:
    1. Outputs Adjacency Matrix (N x N) instead of Feature.
    2. Enforces symmetry to support undirected graph Laplacian diffusion.
    """

    def __init__(self, in_features, num_heads=4, dropout=0.2):
        super(SymmetricGATbuilder, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_features // num_heads

        assert self.head_dim * num_heads == in_features, "in_features must be divisible by num_heads"

        # Linear transformation W
        self.W = nn.Linear(in_features, in_features, bias=False)

        # Attention vector a (for Additive Attention)
        # We use [2 * head_dim] -> [1] mapping
        self.a = nn.Parameter(torch.zeros(num_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.a)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        # h: [B, N, C] (here is Trace)
        B, N, C = h.shape

        # 1. Linear transformation & Multi-head
        # [B, N, C] -> [B, N, Heads, Head_Dim]
        h_prime = self.W(h).view(B, N, self.num_heads, self.head_dim)

        # 2. Build all-pairs (All-pairs)
        # For efficient calculation of Additive Attention: a^T [Wh_i || Wh_j]
        # Equivalent to: a1^T Wh_i + a2^T Wh_j

        # Split parameter a into a_1 and a_2
        a_1 = self.a[:, :self.head_dim]  # [Heads, Head_Dim]
        a_2 = self.a[:, self.head_dim:]  # [Heads, Head_Dim]

        # Calculate a_1 * Wh_i -> [B, N, Heads]
        # einsum: 'bnhd, hd -> bnh'
        e_1 = torch.einsum('bnhd,hd->bnh', h_prime, a_1)
        e_2 = torch.einsum('bnhd,hd->bnh', h_prime, a_2)

        # Broadcasting addition to get N x N matrix
        # [B, N, 1, Heads] + [B, 1, N, Heads] -> [B, N, N, Heads]
        e = e_1.unsqueeze(2) + e_2.unsqueeze(1)
        e = self.leakyrelu(e)

        # 3. [Core Step] Force Symmetry (Symmetrization)
        # Standard GAT is asymmetric, here we take the average: (e_ij + e_ji) / 2
        # Physically this means: i's attention to j and j's attention to i reach a consensus
        e_sym = (e + e.permute(0, 2, 1, 3)) / 2

        # 4. Normalization (Softmax over neighbors)
        # [B, N, N, Heads] -> Mean over Heads -> [B, N, N]
        # Can also do Softmax then Mean, here we do Mean first to save computation
        attention = e_sym.mean(dim=-1)

        # 5. Temperature Scaling (optional, makes graph sparser)
        attention = F.softmax(attention / 0.01, dim=-1)

        return self.dropout(attention)


def conv3x3_snn(in_channels, out_channels):
    """Standard Conv-BN-LIF module"""
    return nn.Sequential(
        layer.SeqToANNContainer(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ),
        MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True)
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
    def __init__(self, node_num, in_channels, out_channels, k_pruning=3, beta_plasticity=0.0, trace_decay=0.6):
        super(STSPBlock, self).__init__()
        self.node_num = node_num
        self.total_nodes = self.node_num
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k_pruning
        self.attn_norm = nn.LayerNorm(out_channels)
        self.gat_builder = SymmetricGATbuilder(out_channels, num_heads=4)
        # === [Core Parameters] ===
        self.trace_decay = trace_decay
        self.beta_param = nn.Parameter(torch.tensor(beta_plasticity), requires_grad=False)

        # === [Feature Mapper] (Unchanged) ===
        self.spatial_pool = layer.SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.feature_transform = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU()
        )

        # === [Node Definition] (Unchanged) ===
        self.module_list = nn.ModuleList([Pooling_Node(self.in_channels, self.out_channels)])
        for _ in range(self.total_nodes - 1):
            self.module_list.append(conv3x3_snn(self.out_channels, self.out_channels))

        self.output_weights = nn.Parameter(torch.ones(self.total_nodes, requires_grad=True))

        # ============================================================
        # [New Modification 1]: Diffusion Control Parameter
        # ============================================================
        # gamma: Controls how much information comes from diffusion vs. retaining original input
        # Initialized to 0.5 for balance
        self.diffusion_gamma = nn.Parameter(torch.tensor(0.5))

    # ============================================================
    # [New Modification 2]: Symmetric Normalized Laplacian Operator Calculation Tool
    # ============================================================
    def compute_diffusion_operator(self, adj):
        """
        Convert any adjacency matrix to a symmetric normalized diffusion operator.
        Formula: P = D^(-1/2) * (A_sym + I) * D^(-1/2)
        This ensures the graph is undirected and the diffusion process is numerically stable (spectral radius <= 1).
        """
        B, N, _ = adj.shape

        # 1. Force Symmetry (Symmetrization): Makes the graph undirected, ensuring energy function existence
        adj_sym = (adj + adj.transpose(1, 2)) / 2

        adj_hat = adj_sym

        # 3. Calculate Degree Matrix D (Degree Matrix)
        degree = adj_hat.sum(dim=2)  # [B, N]

        # 4. Calculate D^(-1/2)
        degree_inv_sqrt = torch.pow(degree + 1e-6, -0.5)  # Add epsilon to prevent division by zero
        D_inv_sqrt_mat = torch.diag_embed(degree_inv_sqrt)  # [B, N, N]

        # 5. Symmetric Normalization: D^(-1/2) * A_hat * D^(-1/2)
        diffusion_op = torch.bmm(torch.bmm(D_inv_sqrt_mat, adj_hat), D_inv_sqrt_mat)

        return diffusion_op

    def get_graph_entropy_loss(self, graph_tensor):
        """
        Calculate graph weight entropy as regularization term.
        Goal: Minimize entropy -> Force graph to become sparse and confident.
        graph_tensor: [B, T, N, N] (Attention Scores)
        """
        # Add epsilon to prevent log(0)
        eps = 1e-8
        # H = - sum( p * log(p) )
        # We want p to approach 0 or 1, so H is minimized
        entropy = -torch.sum(graph_tensor * torch.log(graph_tensor + eps), dim=-1)
        return entropy.mean()

    def prune_topk(self, S, k):
        # (Unchanged)
        top_k_vals, _ = torch.topk(S, k, dim=-1)
        k_th_val = top_k_vals[:, -1].unsqueeze(1)
        mask = (S >= k_th_val).float()
        return S * mask


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

            # Phase 1: Update Input Node & Trace (Unchanged)
            out_0_t = self.module_list[0](x_t)
            feat_0_raw = self.spatial_pool(out_0_t).squeeze(0)
            feat_0_vec = self.feature_transform(feat_0_raw)
            node_traces[:, 0, :] = self.trace_decay * node_traces[:, 0, :] + \
                                   (1 - self.trace_decay) * feat_0_vec

            # Phase 2: Structural Plasticity (Unchanged)
            # alpha_t = self.compute_trace_attention(node_traces)
            alpha_t = self.gat_builder(node_traces)
            S = beta * S.detach() + (1 - beta) * alpha_t

            if return_graph:
                graph_snapshots.append(S.detach())

            S_pruned = torch.stack([self.prune_topk(S[b], self.k) for b in range(B)])

            # ============================================================
            # Phase 3: [Core Modification] Graph Propagation Based on Laplacian Diffusion
            # ============================================================

            # 1. Prepare Source Signal (Source Signal)
            out_flat = out_0_t.squeeze(0).view(B, -1)  # [B, Feature_Dim]
            X_source = torch.zeros(B, N, out_flat.shape[1], device=x.device)
            X_source[:, 0, :] = out_flat  # Only node 0 has input, acting as "heat source"

            # 2. Calculate Diffusion Operator (Diffusion Operator)
            # Here we call the new function to convert directed S_pruned to undirected normalized P
            # This step corresponds to the theory: Minimizing Dirichlet Energy

            # diffusion_op = self.compute_diffusion_operator(S_pruned)
            diffusion_op = self.compute_diffusion_operator(S_pruned)
            #  Will correspond to a beautiful theory.
            H_diffused = torch.bmm(diffusion_op, X_source)
            H_diffused = torch.bmm(diffusion_op, H_diffused)

            # 4. Source Injection & Fusion (Source Injection & Fusion)
            # Formula: H_out = (1 - gamma) * X_source + gamma * H_diffused
            # Physical meaning: Ensure source node information is retained, while allowing information to diffuse across the network

            mat_t = H_diffused
            # ============================================================
            # Phase 3 End
            # ============================================================

            out_list_t = [out_0_t]
            feats_instant_list = [feat_0_vec]

            for i in range(1, N):
                # Reshape back to [B, C, H, W] (Unchanged)
                in_i_t = mat_t[:, i, :].reshape(B, self.out_channels, out_0_t.shape[-2], out_0_t.shape[-1]).unsqueeze(0)

                out_i_t = self.module_list[i](in_i_t)
                out_list_t.append(out_i_t)

                feat_i_raw = self.spatial_pool(out_i_t).squeeze(0)
                feat_i_vec = self.feature_transform(feat_i_raw)
                feats_instant_list.append(feat_i_vec)

            # Phase 4: Update Traces & Aggregate (Unchanged)
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
# 3. Overall Model (DVS Gesture Adaptation Version)
# ==========================================

class Model(nn.Module):
    def __init__(self, node_num=7, in_channels=2, out_channels=64, num_classes=11):
        super(Model, self).__init__()

        # Stem Layer: Process DVS raw input (2 channels: On/Off Events)
        self.conv1 = nn.Sequential(
            layer.SeqToANNContainer(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
            ),
            MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True)
        )
        self.pool1 = layer.SeqToANNContainer(nn.AvgPool2d(2, 2))

        # Core Dynamic Graph Module
        # Note: beta_plasticity=0.0 lets Trace dominate
        self.stsp_block = STSPBlock(node_num, out_channels, out_channels,
                                    k_pruning=3, beta_plasticity=0.2, trace_decay=0.6)

        self.stsp_block2 = STSPBlock(node_num, out_channels, out_channels,
                                     k_pruning=3, beta_plasticity=0.2, trace_decay=0.6)

        self.stsp_block3 = STSPBlock(node_num, out_channels, out_channels,
                                     k_pruning=3, beta_plasticity=0.2, trace_decay=0.6)

        self.pool2 = layer.SeqToANNContainer(nn.AvgPool2d(2, 2))

        self.flatten = nn.Flatten(2)
        self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, x, return_graph=False):
        # x: [T, B, 2, 128, 128] (assumed)

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
    # Simple test stub
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # DVS Gesture input simulation: [T=16, B=2, C=2, H=32, W=32]
    x = torch.rand((16, 2, 2, 128, 128)).to(device)
    x = (x > 0.8).float()  # Simulate sparse spikes

    net = Model(node_num=7, in_channels=2, out_channels=112, num_classes=10).to(device)

    # Here 8*8 needs to be adjusted according to the input above, 32/2/2 = 8, so 8*8 in the code is correct

    y, graph = net(x, return_graph=True)
    print(f"Output Shape: {y.shape}")
    print(f"Graph Shape: {graph.shape} (Batch, Time, Node, Node)")

    # Check if Graph is all zeros (common DVS pitfall)
    print(f"Graph Mean: {graph.mean().item()}")
    print(f"Graph Std: {graph.std().item()}")