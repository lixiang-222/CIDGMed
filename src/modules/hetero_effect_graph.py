import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import RGCNConv


class hetero_effect_graph(nn.Module):
    def __init__(self, in_channels, out_channels, device, levels=5):
        super(hetero_effect_graph, self).__init__()

        self.device = device

        # 等级数量，用于划分不同权重级别的边 最后加1代表生成一种虚拟的边
        self.levels = levels + 1

        # 边类型映射字典
        self.edge_type_mapping = {}
        self.initialize_edge_type_mapping()

        # 定义两个RGCN卷积层
        self.conv1 = RGCNConv(in_channels, out_channels, self.levels)
        self.conv2 = RGCNConv(out_channels, out_channels, self.levels)

    def initialize_edge_type_mapping(self):
        # 分配整数值给每种边类型
        j = 0
        for i in range(self.levels + 1):
            edge_type = ('Mole', f'connected__{i}', 'Entity')
            self.edge_type_mapping[edge_type] = j
            j += 1


    def create_hetero_graph(self, emb_entity, emb_mole, entity_mole_weight):
        # 创建异构图数据结构
        data = HeteroData()

        # 分配节点嵌入
        data['Entity'].x = emb_entity.squeeze(0)
        data['Mole'].x = emb_mole.squeeze(0)

        # 如果全部是0向量不用分层
        if np.all(entity_mole_weight == 0):
            src = torch.zeros(entity_mole_weight.size, dtype=torch.int64)
            dst = torch.arange(0, entity_mole_weight.size, dtype=torch.int64)
            edge_index = torch.stack([src, dst])
            data['Mole', f'connected__{0}', 'Entity'].edge_index = edge_index
            print('如果全部是0向量不用分层')
        else:
            # 根据权重为关系分配不同的关系类型
            for i in range(1, self.levels):
                mask = (entity_mole_weight > (i / self.levels)) & \
                       (entity_mole_weight <= ((i + 1) / self.levels))
                edge_index = torch.from_numpy(np.vstack(mask.nonzero()))

                if edge_index.size(0) > 0:
                    # 不需要具体的权重，知道属于第几类边就可以了
                    edge_index = edge_index.flip([0])
                    data['Mole', f'connected__{i}', 'Entity'].edge_index = edge_index

        return data

    def hetero_to_homo(self, data):
        # 统一编码所有节点
        entity_offset = 0
        mole_offset = entity_offset + data['Entity'].x.size(0)

        # 合并所有节点特征，x_all是所有节点的嵌入
        x_all = torch.cat([data['Entity'].x, data['Mole'].x], dim=0)

        # 创建整张图的edge_index和edge_type
        edge_index_list = []
        edge_type_list = []

        # range+1为了适配虚拟类
        for i in range(self.levels):
            key = ('Mole', f'connected__{i}', 'Entity')
            if key in data.edge_types:
                src, dst = data[key].edge_index
                edge_index_list.append(torch.stack([src + mole_offset, dst + entity_offset], dim=0))
                edge_type_list.append(torch.full((len(src),), self.edge_type_mapping[key]))

        # Concatenate edge_index from different edge types
        edge_index = torch.cat(edge_index_list, dim=1).to(self.device)

        # Concatenate edge_type from different edge types
        edge_type = torch.cat(edge_type_list, dim=0).to(self.device)

        return x_all, edge_index, edge_type

    def forward(self, emb_entity, emb_mole, entity_mole_weights):
        # 创建异构图
        data = self.create_hetero_graph(emb_entity, emb_mole, entity_mole_weights)

        # 从异构图转换到同构图
        x, edge_index, edge_type = self.hetero_to_homo(data)

        # 卷积
        out1 = self.conv1(x, edge_index, edge_type)
        out1 = F.relu(out1)
        out = self.conv2(out1, edge_index, edge_type)

        # 根据偏移量切割张量，分解出每种类型的嵌入
        entity_offset = 0
        mole_offset = entity_offset + data['Entity'].x.size(0)

        out_emb_entity = out[entity_offset:mole_offset]
        out_emb_mole = out[mole_offset:]  # 理论上不需要了

        return out_emb_entity.unsqueeze(0)


if __name__ == '__main__':
    # 示例数据（错的）
    torch.manual_seed(1203)
    np.random.seed(2048)

    diag_emb = torch.randn(5, 8)
    proc_emb = torch.randn(3, 8)
    med_emb = torch.randn(4, 8)

    diag_med_weights = np.random.rand(5, 4)
    proc_med_weights = np.random.rand(3, 4)

    # 创建模型并计算输出
    model = hetero_effect_graph(8, 8, torch.device("cpu"))
    out = model(diag_emb, proc_emb, med_emb, diag_med_weights, proc_med_weights)
