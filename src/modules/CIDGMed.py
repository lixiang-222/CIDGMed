import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINConv

from .hetero_effect_graph import hetero_effect_graph
from .homo_relation_graph import homo_relation_graph


class CausaltyReview(nn.Module):
    def __init__(self, casual_graph, num_diag, num_proc, num_med):
        super(CausaltyReview, self).__init__()

        self.num_med = num_med
        self.c1 = casual_graph
        diag_med_high = casual_graph.get_threshold_effect(0.97, "Diag", "Med")
        diag_med_low = casual_graph.get_threshold_effect(0.90, "Diag", "Med")
        proc_med_high = casual_graph.get_threshold_effect(0.97, "Proc", "Med")
        proc_med_low = casual_graph.get_threshold_effect(0.90, "Proc", "Med")
        self.c1_high_limit = nn.Parameter(torch.tensor([diag_med_high, proc_med_high]))  # 选用的97%
        self.c1_low_limit = nn.Parameter(torch.tensor([diag_med_low, proc_med_low]))  # 选用的90%
        self.c1_minus_weight = nn.Parameter(torch.tensor(0.01))
        self.c1_plus_weight = nn.Parameter(torch.tensor(0.01))

    def forward(self, pre_prob, diags, procs):
        reviewed_prob = pre_prob.clone()

        for m in range(self.num_med):
            max_cdm = 0.0
            max_cpm = 0.0
            for d in diags:
                cdm = self.c1.get_effect(d, m, "Diag", "Med")
                max_cdm = max(max_cdm, cdm)
            for p in procs:
                cpm = self.c1.get_effect(p, m, "Proc", "Med")
                max_cpm = max(max_cpm, cpm)

            if max_cdm < self.c1_low_limit[0] and max_cpm < self.c1_low_limit[1]:
                reviewed_prob[0, m] -= self.c1_minus_weight
            elif max_cdm > self.c1_high_limit[0] or max_cpm > self.c1_high_limit[1]:
                reviewed_prob[0, m] += self.c1_plus_weight

        return reviewed_prob


class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GIN, self).__init__()
        # 定义一个两层的MLP（多层感知机）作为GIN中的聚合函数
        nn = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        # 使用GINConv来定义GIN层
        self.conv = GINConv(nn)

    def forward(self, x, edge_index, weights):
        # 应用GIN层
        x = self.conv(x, edge_index)

        # 确保weights的形状与x匹配
        weights = weights.unsqueeze(dim=-1)

        # 使用weights对每个节点嵌入进行加权
        x = x * weights

        # 对所有节点嵌入进行加和
        x_sum = x.sum(dim=0)
        x_sum = x_sum.unsqueeze(dim=0).unsqueeze(dim=0)

        return x_sum


class CIDGMed(torch.nn.Module):
    def __init__(
            self,
            causal_graph,
            mole_relevance,
            tensor_ddi_adj,
            emb_dim,
            voc_size,
            dropout,
            device=torch.device('cpu'),
    ):
        super(CIDGMed, self).__init__()
        self.device = device
        self.emb_dim = emb_dim

        # Embedding of all entities
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(voc_size[0], emb_dim),
            torch.nn.Embedding(voc_size[1], emb_dim),
            torch.nn.Embedding(voc_size[2], emb_dim),  # 这里不用embedding【2】
            torch.nn.Embedding(voc_size[3], emb_dim)
        ])

        if dropout > 0 and dropout < 1:
            self.rnn_dropout = torch.nn.Dropout(p=dropout)
        else:
            self.rnn_dropout = torch.nn.Sequential()

        self.causal_graph = causal_graph

        self.mole_relevance = mole_relevance

        # 这里只是分子与药物的相关性
        self.mole_med_relevance = torch.tensor(mole_relevance[2])
        self.mole_med_weights = nn.Parameter(torch.ones(mole_relevance[2].shape[1]))

        self.gin_model = GIN(emb_dim, emb_dim)

        self.homo_graph = nn.ModuleList([
            homo_relation_graph(emb_dim, device),
            homo_relation_graph(emb_dim, device),
            homo_relation_graph(emb_dim, device)
        ])

        self.hetero_graph = torch.nn.ModuleList([
            hetero_effect_graph(emb_dim, emb_dim, device),
            hetero_effect_graph(emb_dim, emb_dim, device),
            hetero_effect_graph(emb_dim, emb_dim, device)
        ])

        # Isomeric and isomeric addition parameters
        self.rho = nn.Parameter(torch.ones(3, 2))

        self.seq_encoders = torch.nn.ModuleList([
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True),
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True),
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True)
        ])

        # Convert patient information to drug score
        self.query = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 6, voc_size[2])
        )

        self.review = CausaltyReview(self.causal_graph, voc_size[0], voc_size[1], voc_size[2])

        self.tensor_ddi_adj = tensor_ddi_adj
        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)

    def create_graph_data(self, molecule_embeddings):
        num_molecules = len(molecule_embeddings)

        # 构建全连接图的边索引
        source = []
        target = []
        for i in range(num_molecules):
            for j in range(num_molecules):
                if i != j:  # 排除自环
                    source.append(i)
                    target.append(j)

        edge_index = torch.tensor([source, target], dtype=torch.long)

        # 将分子的嵌入向量堆叠起来，作为图的节点特征
        # x = torch.stack(molecule_embeddings)

        # 创建图数据对象
        data = Data(x=molecule_embeddings, edge_index=edge_index)

        return data

    def med_embedding(self, idx_list, emb_mole):
        emb_mole = emb_mole.squeeze(0)
        # 存储所有药物的嵌入表示
        all_drug_embeddings = []

        # 遍历所有药物的索引
        for idx in idx_list:
            # 获取药物与所有分子之间的相关性
            relevance = self.mole_med_relevance[idx, :].to(self.device)
            # 创建一个掩码，标识非零元素的位置
            mask = relevance != 0
            # 将零元素设置为负无穷大，以便在softmax中保持其值为零
            relevance_masked = relevance.masked_fill(~mask, -float('inf'))
            # 对掩码后的relevance使用softmax进行归一化
            relevance_normalized = F.softmax(relevance_masked, dim=0)

            # 找到每个药物与之相关的分子的位置
            relevant_molecule_indices = torch.nonzero(relevance_normalized, as_tuple=True)[0]
            relevant_molecule_embeddings = emb_mole[relevant_molecule_indices]
            weights = self.mole_med_weights[relevant_molecule_indices]
            weights_normalized = F.softmax(weights, dim=0)

            # 使用GIN构建分子图网络并获取药物表示
            graph_data = self.create_graph_data(relevant_molecule_embeddings)

            # 使用GIN模型得到药物的嵌入表示
            drug_embedding = self.gin_model(graph_data.x.to(self.device), graph_data.edge_index.to(self.device), weights_normalized)
            all_drug_embeddings.append(drug_embedding)

        # 将所有药物的嵌入表示按顺序堆叠在一起
        all_drug_embeddings = torch.cat(all_drug_embeddings, dim=1)
        return all_drug_embeddings

    def forward(self, patient_data):
        seq_diag, seq_proc, seq_med = [], [], []
        for adm_id, adm in enumerate(patient_data):
            # 获取所有分子的嵌入
            num_moles = self.embeddings[3].num_embeddings
            idx_mole = torch.arange(num_moles).to(self.device)
            emb_mole = self.embeddings[3](idx_mole).unsqueeze(0)

            idx_diag = torch.LongTensor(adm[0]).to(self.device)
            idx_proc = torch.LongTensor(adm[1]).to(self.device)
            emb_diag = self.rnn_dropout(self.embeddings[0](idx_diag)).unsqueeze(0)
            emb_proc = self.rnn_dropout(self.embeddings[1](idx_proc)).unsqueeze(0)

            relevance_diag = self.mole_relevance[0][adm[0], :]
            emb_diag1 = self.hetero_graph[0](emb_diag, emb_mole, relevance_diag)

            relevance_proc = self.mole_relevance[1][adm[1], :]
            emb_proc1 = self.hetero_graph[1](emb_proc, emb_mole, relevance_proc)

            graph_diag = self.causal_graph.get_graph(adm[3], "Diag")
            graph_proc = self.causal_graph.get_graph(adm[3], "Proc")
            emb_diag2 = self.homo_graph[0](graph_diag, emb_diag1)
            emb_proc2 = self.homo_graph[1](graph_proc, emb_proc1)

            # 对上次药物包的学习
            if adm == patient_data[0]:
                emb_med2 = torch.zeros(1, 1, self.emb_dim).to(self.device)
            else:
                adm_last = patient_data[adm_id - 1]
                emb_med1 = self.rnn_dropout(self.med_embedding(adm_last[2], emb_mole))

                med_graph = self.causal_graph.get_graph(adm_last[3], "Med")
                emb_med2 = self.homo_graph[2](med_graph, emb_med1)

            seq_diag.append(torch.sum(emb_diag2, keepdim=True, dim=1))
            seq_proc.append(torch.sum(emb_proc2, keepdim=True, dim=1))
            seq_med.append(torch.sum(emb_med2, keepdim=True, dim=1))

        seq_diag = torch.cat(seq_diag, dim=1)
        seq_proc = torch.cat(seq_proc, dim=1)
        seq_med = torch.cat(seq_med, dim=1)
        output_diag, hidden_diag = self.seq_encoders[0](seq_diag)
        output_proc, hidden_proc = self.seq_encoders[1](seq_proc)
        output_med, hidden_med = self.seq_encoders[2](seq_med)
        seq_repr = torch.cat([hidden_diag, hidden_proc, hidden_med], dim=-1)
        last_repr = torch.cat([output_diag[:, -1], output_proc[:, -1], output_med[:, -1]], dim=-1)
        patient_repr = torch.cat([seq_repr.flatten(), last_repr.flatten()])

        score = self.query(patient_repr).unsqueeze(0)

        # 用来review的
        score = self.review(score, patient_data[-1][0], patient_data[-1][1])

        neg_pred_prob = torch.sigmoid(score)
        neg_pred_prob = torch.matmul(neg_pred_prob.t(), neg_pred_prob)
        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        return score, batch_neg
