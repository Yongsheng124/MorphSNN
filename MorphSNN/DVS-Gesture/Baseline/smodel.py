import sys
import torch
import torch.nn as nn
from spikingjelly.clock_driven import layer, encoding
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode
from thop import profile
from graph import RandomGraph


def conv3x3(in_channels, out_channels):
    return nn.Sequential(
        layer.SeqToANNContainer(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ),
        MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True)
    )


class SEWBlock(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(SEWBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv3x3(in_channels, mid_channels),
        )

    def forward(self, x: torch.Tensor):
        out = self.conv1(x)
        return out


class Unit(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(Unit, self).__init__()
        self.sewblock = SEWBlock(in_channels, mid_channels)

    def forward(self, x):
        out = self.sewblock(x)
        return out


class Pooling_Unit(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(Pooling_Unit, self).__init__()
        self.sewblock = SEWBlock(in_channels, mid_channels)
        self.avgp = layer.SeqToANNContainer(nn.AvgPool2d(2, 2))

    def forward(self, x):
        out = self.sewblock(x)
        out = self.avgp(out)
        return out


class Pooling_Node(nn.Module):
    def __init__(self, in_degree, in_channels, out_channels):
        super(Pooling_Node, self).__init__()
        self.in_degree = in_degree
        if len(self.in_degree) > 1:
            self.weights = nn.Parameter(torch.ones(len(self.in_degree), requires_grad=True))
            # 如果一个结点的入度大于1，就给他赋予可训练的权重参数，长度为入度数，值为1
        else:
            self.weights = torch.ones(1)
        self.pooling_unit = Pooling_Unit(in_channels, out_channels)
        # unit对应的即是文章中所说的Transformation

    def forward(self, *input):
        if len(self.in_degree) > 1:
            input = list(input)
            x = (input[0] * torch.sigmoid(self.weights[0]))
            for index in range(1, len(input)):
                x += (input[index] * torch.sigmoid(self.weights[index]))
            out = self.pooling_unit(x)
        else:

            out = self.pooling_unit(input[0])
        return out


class Node(nn.Module):
    def __init__(self, in_degree, in_channels, out_channels):
        super(Node, self).__init__()
        self.in_degree = in_degree
        if len(self.in_degree) > 1:
            self.weights = nn.Parameter(torch.ones(len(self.in_degree), requires_grad=True))
            # 如果一个结点的入度大于1，就给他赋予可训练的权重参数，长度为入度数，值为1
        else:
            self.weights = torch.ones(1)
        self.unit = Unit(in_channels, out_channels)
        # unit对应的即是文章中所说的Transformation

    def forward(self, *input):
        if len(self.in_degree) > 1:
            input = list(input)
            x = (input[0] * torch.sigmoid(self.weights[0]))
            for index in range(1, len(input)):
                x += (input[index] * torch.sigmoid(self.weights[index]))
            out = self.unit(x)
        else:

            out = self.unit(input[0])
        return out


class RandWire(nn.Module):
    def __init__(self, node_num, p, in_channels, out_channels, graph_mode, output_dir):
        super(RandWire, self).__init__()
        self.node_num = node_num
        self.p = p
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.graph_mode = graph_mode
        self.memory = {}  # memory 是一个字典统计每一个模块的输出
        self.wire_weights = {}
        self.output_dir = output_dir

        graph_node = RandomGraph(self.node_num, self.p, graph_mode=graph_mode)
        graph = graph_node.make_graph()
        self.nodes, self.in_edges = graph_node.get_graph_info(graph)
        self.in_edges = {0: [], 1: [0], 2: [0, 1], 3: [0, 1, 2], 4: [0, 1, 2, 3], 5: [0, 1, 2, 3, 4],
                         6: [0, 1, 2, 3, 4, 5]}
        print(self.in_edges)

        self.module_list = nn.ModuleList(
            [Pooling_Node(self.in_edges[0], self.in_channels, self.out_channels)])
        for node in self.nodes:
            if node > 0:
                self.module_list.append(Node(self.in_edges[node], self.out_channels, self.out_channels))

        # define the rest Node
        # self.module_list.extend(
        #     [Node(self.in_edges[node], self.out_channels, self.out_channels) for node in self.nodes if node > 0])
        # 把每个节点模块都累积到moduleList这个容器里面

    def forward(self, x):
        # memory 是一个字典统计每一个模块的输出
        # start vertex
        out = self.module_list[0].forward(x)
        # print(out.shape)
        self.memory[0] = out
        # memory保存Node 0的输出

        # 剩余的中间Node， 现在没有管最后一个Node
        for node in range(1, len(self.nodes)):
            # print(node, self.in_edges[node][0], self.in_edges[node])
            if len(self.in_edges[node]) > 1:
                # 如果Node的入度是大于0的
                # 他们的输出应该是他所有入度节点的输出然后送进自己的forward里（加权在自己forward里面管）
                out = self.module_list[node].forward(*[self.memory[in_vertex] for in_vertex in self.in_edges[node]])
            else:

                out = self.module_list[node].forward(self.memory[self.in_edges[node][0]])
            # 保存到字典里面
            self.memory[node] = out

        out = self.memory[self.node_num + 1]
        return out

    def monitor(self):
        return self.memory

    def get_weights(self):
        return self.wire_weights


class Model(nn.Module):
    def __init__(self, node_num, p, in_channels, out_channels, graph_mode, output_dir):
        super(Model, self).__init__()
        self.node_num = node_num
        self.p = p
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.graph_mode = graph_mode
        self.output_dir = output_dir

        self.num_classes = 11
        self.dropout_rate = 0.2
        self.memory = {}
        self.conv1 = nn.Sequential(
            layer.SeqToANNContainer(
                nn.Conv2d(in_channels=2, out_channels=self.out_channels, kernel_size=3, padding=1, stride=1,
                          bias=False),
                nn.BatchNorm2d(self.out_channels),
            ),
            MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True))

        self.randwire1 = RandWire(self.node_num, self.p, self.in_channels, self.out_channels, self.graph_mode,
                                  self.output_dir)

        self.randwire2 = RandWire(self.node_num, self.p, self.in_channels, self.out_channels, self.graph_mode,
                                  self.output_dir)

        self.randwire3 = RandWire(self.node_num, self.p, self.in_channels, self.out_channels, self.graph_mode,
                                  self.output_dir)

        self.pool2 = layer.SeqToANNContainer(nn.AvgPool2d(2, 2))

        self.flatten = nn.Flatten(2)

        self.CIFAR_classifier = nn.Linear(self.out_channels, self.num_classes, bias=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool2(out)
        out = self.randwire1(out)
        out = self.pool2(out)
        out = self.randwire2(out)
        out = self.pool2(out)
        out = self.randwire3(out)
        out = self.pool2(out)
        out = self.flatten(out)
        out = out.mean(0)
        out = self.CIFAR_classifier(out)
        return out

    def monitor(self):
        randwire_memory = self.randwire.monitor()
        self.memory.update(randwire_memory)
        return self.memory

    def get_weights(self):
        return self.randwire.get_weights()


def cal_firing_rate(spike_seq):
    return spike_seq.flatten().mean(0)


if __name__ == "__main__":
    x2 = torch.round(torch.rand((5, 1, 2, 128, 128))).to("cuda")
    net = Model(5, 0.75, 32, 32, 'WS', './test').to(device="cuda")
    # print(net)
    print(net(x2).shape)

    Flops, params = profile(net, inputs=(x2,))
    print('Flops: %.4fG' % (Flops / 1e9))  # 将 FLOPs 转换为 GFLOPs
    print('params参数量: % .4fM' % (params / 1000000))
