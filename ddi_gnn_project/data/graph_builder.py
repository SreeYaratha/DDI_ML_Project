"""
Build graph dataset for GNN training.
"""
import warnings
from typing import Dict, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data

warnings.filterwarnings("ignore")


class DDIGraphBuilder:
    """
    Builds a heterogeneous graph from the DDI interaction table and drug features.
    """

    def __init__(
        self,
        ddi_path: str,
        drug_features_path: str,
        drug1_col: str,
        drug2_col: str,
        interaction_col: str,
        min_interactions: int = 0,
        binary_labels: bool = True,
        max_classes: int = 64,
    ):
        self.ddi_path = ddi_path
        self.drug_features_path = drug_features_path
        self.drug1_col = drug1_col
        self.drug2_col = drug2_col
        self.interaction_col = interaction_col
        self.min_interactions = min_interactions
        self.binary_labels = binary_labels
        self.max_classes = max_classes

        self.graph = None
        self.pyg_data = None
        self.label_encoder = LabelEncoder()
        self.drug_to_idx: Dict[str, int] = {}

    def load_and_build(self) -> Data:
        """
        Load source CSVs and construct a torch-geometric Data object.
        """
        ddi_df = pd.read_csv(self.ddi_path)
        drug_features_df = pd.read_csv(self.drug_features_path)

        ddi_df = ddi_df.rename(
            columns={
                self.drug1_col: "drug1",
                self.drug2_col: "drug2",
                self.interaction_col: "interaction",
            }
        )

        required_cols = {"drug1", "drug2", "interaction"}
        missing = required_cols - set(ddi_df.columns)
        if missing:
            raise ValueError(f"DDI file missing required columns: {missing}")

        # Filter to drugs that have features
        # The processed feature file stores the DDI name under 'ddi_name'
        name_to_features = drug_features_df.set_index("ddi_name")

        ddi_df = ddi_df[
            ddi_df["drug1"].isin(name_to_features.index)
            & ddi_df["drug2"].isin(name_to_features.index)
        ].reset_index(drop=True)

        if len(ddi_df) == 0:
            raise ValueError("No DDI rows left after aligning with drug features.")

        # Optionally prune very low-degree drugs
        if self.min_interactions > 0:
            counts = (
                ddi_df["drug1"].value_counts()
                + ddi_df["drug2"].value_counts()
            ).fillna(0)
            keep_drugs = counts[counts >= self.min_interactions].index
            ddi_df = ddi_df[
                ddi_df["drug1"].isin(keep_drugs) & ddi_df["drug2"].isin(keep_drugs)
            ].reset_index(drop=True)

        unique_drugs = pd.Index(
            pd.concat([ddi_df["drug1"], ddi_df["drug2"]]).unique()
        )
        self.drug_to_idx = {drug: i for i, drug in enumerate(unique_drugs)}

        # Build graph
        self.graph = nx.Graph()
        for drug in unique_drugs:
            self.graph.add_node(self.drug_to_idx[drug], drug_name=drug)

        # Encode interaction labels
        if self.binary_labels:
            ddi_df = ddi_df.assign(label=1)
            self.label_encoder = None
        else:
            counts = ddi_df["interaction"].value_counts()
            keep = counts.nlargest(self.max_classes - 1).index
            ddi_df["interaction_capped"] = ddi_df["interaction"].where(
                ddi_df["interaction"].isin(keep), other="Other"
            )
            y = self.label_encoder.fit_transform(ddi_df["interaction_capped"])
            ddi_df = ddi_df.assign(label=y)

        edge_list = []
        edge_labels = []
        for _, row in ddi_df.iterrows():
            src = self.drug_to_idx[row["drug1"]]
            dst = self.drug_to_idx[row["drug2"]]
            label = int(row["label"])

            edge_list.append((src, dst))
            edge_list.append((dst, src))
            edge_labels.append(label)
            edge_labels.append(label)
            self.graph.add_edge(src, dst, label=label)

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_labels, dtype=torch.long)

        # Build node feature matrix
        numeric_cols = [
            c
            for c in drug_features_df.columns
            if c not in ["drugbank_id", "ddi_name"]
            and pd.api.types.is_numeric_dtype(drug_features_df[c])
        ]
        if not numeric_cols:
            # Fallback to random embeddings if no numeric features exist
            num_features = 128
            x = torch.randn(len(unique_drugs), num_features, dtype=torch.float)
        else:
            subset = name_to_features.loc[unique_drugs, numeric_cols]
            subset = subset.fillna(subset.mean())
            x = torch.tensor(subset.values, dtype=torch.float)

        self.pyg_data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(unique_drugs),
        )
        return self.pyg_data

    def get_statistics(self) -> Dict[str, float]:
        """Compute simple NetworkX statistics for the built graph."""
        if self.graph is None:
            return {}

        stats = {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "average_degree": (
                sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes()
            ),
            "density": nx.density(self.graph),
            "connected_components": nx.number_connected_components(self.graph),
        }
        return stats

    def get_label_encoder(self) -> LabelEncoder:
        """Return the fitted label encoder (None when binary_labels=True)."""
        return self.label_encoder

    def get_index_mapping(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Return drug<->index mappings."""
        idx_to_drug = {v: k for k, v in self.drug_to_idx.items()}
        return self.drug_to_idx, idx_to_drug
