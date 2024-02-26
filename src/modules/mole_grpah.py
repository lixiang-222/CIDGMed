import torch
import torch.nn as nn


class MoleMedGraph(nn.Module):
    def __init__(self, relation_matrix):
        super(MoleMedGraph, self).__init__()
        self.relation_matrix = nn.Parameter(torch.FloatTensor(relation_matrix))

    def get_relevance(self, med, mole):
        return self.relation_matrix[med][mole]
