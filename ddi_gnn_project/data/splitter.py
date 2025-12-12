"""
Utilities for splitting graphs into cold-start train/val/test sets.
"""
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from sklearn.model_selection import train_test_split


@dataclass
class SplitResult:
    train_edges: torch.Tensor
    train_labels: torch.Tensor
    val_edges: torch.Tensor
    val_labels: torch.Tensor
    test_edges: torch.Tensor
    test_labels: torch.Tensor


class ColdStartSplitter:
    """
    Splits edges by holding out a set of nodes (cold-start).
    """

    def __init__(self, test_ratio: float, val_ratio: float, seed: int):
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.seed = seed

    def split(self, graph_data: torch.Tensor, cold_start: bool = True) -> SplitResult:
        edge_index = graph_data.edge_index.t().cpu().numpy()
        edge_labels = graph_data.edge_attr.cpu().numpy()
        num_nodes = graph_data.num_nodes

        # Build edge tuples; keep both directions to preserve signal symmetry
        edges = np.column_stack([edge_index[:, 0], edge_index[:, 1]])

        if cold_start:
            node_indices = np.arange(num_nodes)
            train_nodes, test_nodes = train_test_split(
                node_indices,
                test_size=self.test_ratio,
                random_state=self.seed,
            )
            train_nodes = set(train_nodes.tolist())
            test_nodes = set(test_nodes.tolist())

            train_mask = []
            test_mask = []
            for src, dst in edges:
                if src in train_nodes and dst in train_nodes:
                    train_mask.append(True)
                    test_mask.append(False)
                elif src in test_nodes or dst in test_nodes:
                    train_mask.append(False)
                    test_mask.append(True)
                else:
                    # default to train if neither set captured (should not happen)
                    train_mask.append(True)
                    test_mask.append(False)
        else:
            train_mask = np.zeros(len(edges), dtype=bool)
            test_mask = np.zeros(len(edges), dtype=bool)
            split_mask = np.random.default_rng(self.seed).uniform(size=len(edges)) >= self.test_ratio
            train_mask[:] = split_mask
            test_mask[:] = ~split_mask

        train_edges = torch.tensor(edges[train_mask], dtype=torch.long)
        train_labels = torch.tensor(edge_labels[train_mask], dtype=torch.long)
        test_edges = torch.tensor(edges[test_mask], dtype=torch.long)
        test_labels = torch.tensor(edge_labels[test_mask], dtype=torch.long)

        # Derive validation split from the training set
        if len(train_edges) == 0:
            raise ValueError("No training edges after split; check data or split ratios.")

        val_size = max(1, int(len(train_edges) * self.val_ratio))
        perm = torch.randperm(len(train_edges), generator=torch.Generator().manual_seed(self.seed))
        val_idx = perm[:val_size]
        keep_idx = perm[val_size:]

        val_edges = train_edges[val_idx]
        val_labels = train_labels[val_idx]
        train_edges = train_edges[keep_idx]
        train_labels = train_labels[keep_idx]

        return SplitResult(
            train_edges=train_edges,
            train_labels=train_labels,
            val_edges=val_edges,
            val_labels=val_labels,
            test_edges=test_edges,
            test_labels=test_labels,
        )
