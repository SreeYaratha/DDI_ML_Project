"""
Graph Neural Network architectures for DDI prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv


class EdgeGNN(nn.Module):
    """
    GCN-based encoder with an edge classifier head.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3):
        super().__init__()
        layers = [GCNConv(input_dim, hidden_dim)]
        layers += [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers - 1)]
        self.layers = nn.ModuleList(layers)

        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in layers])
        self.dropout = nn.Dropout(dropout)

        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x, edge_index, drug_pairs):
        h = x
        for conv, norm in zip(self.layers, self.norms):
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)
            h = self.dropout(h)

        drug1_emb = h[drug_pairs[:, 0]]
        drug2_emb = h[drug_pairs[:, 1]]
        combined = torch.cat([drug1_emb, drug2_emb], dim=1)
        return self.edge_classifier(combined)


class GATDDI(nn.Module):
    """
    Two-layer Graph Attention Network for DDI edge classification.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, heads=4, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, edge_index, drug_pairs):
        h = self.conv1(x, edge_index)
        h = F.elu(h)
        h = self.dropout(h)

        h = self.conv2(h, edge_index)
        h = F.elu(h)

        drug1_emb = h[drug_pairs[:, 0]]
        drug2_emb = h[drug_pairs[:, 1]]
        combined = torch.cat([drug1_emb, drug2_emb], dim=1)
        return self.edge_predictor(combined)


class LinkPredictionGNN(nn.Module):
    """
    Simplified binary link prediction GCN baseline.
    """

    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        layers = [GCNConv(input_dim, hidden_dim)]
        layers += [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers - 1)]
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)

        self.link_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, edge_index, drug_pairs):
        h = x
        for conv in self.layers:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = self.dropout(h)

        drug1_emb = h[drug_pairs[:, 0]]
        drug2_emb = h[drug_pairs[:, 1]]
        combined = torch.cat([drug1_emb, drug2_emb], dim=1)
        return self.link_head(combined)
