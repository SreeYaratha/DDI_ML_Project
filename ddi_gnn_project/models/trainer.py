"""
Training and evaluation utilities for GNN DDI models.
"""
import copy
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .gnn import EdgeGNN, GATDDI, LinkPredictionGNN
from utils.metrics import binary_metrics, multiclass_metrics


class GNNTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _init_model(self, input_dim: int, num_classes: int) -> nn.Module:
        model_type = self.config.MODEL_TYPE.lower()
        if model_type == "edgegnn":
            model = EdgeGNN(
                input_dim=input_dim,
                hidden_dim=self.config.HIDDEN_DIM,
                output_dim=num_classes,
                num_layers=self.config.NUM_LAYERS,
                dropout=self.config.DROPOUT,
            )
        elif model_type == "gat":
            model = GATDDI(
                input_dim=input_dim,
                hidden_dim=self.config.HIDDEN_DIM,
                output_dim=num_classes,
                heads=self.config.HEADS,
                dropout=self.config.DROPOUT,
            )
        else:
            model = LinkPredictionGNN(
                input_dim=input_dim,
                hidden_dim=self.config.HIDDEN_DIM,
                num_layers=self.config.NUM_LAYERS,
                dropout=self.config.DROPOUT,
            )
        return model.to(self.device)

    def train(
        self,
        graph_data: torch.Tensor,
        splits,
        num_classes: int,
    ) -> Dict:
        """Train the chosen GNN model."""
        model = self._init_model(graph_data.x.shape[1], num_classes)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
        )

        if num_classes > 2:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCEWithLogitsLoss()

        best_state = None
        best_val = -1
        patience_counter = 0
        history: List[Dict[str, float]] = []

        graph_data = graph_data.to(self.device)

        train_edges = splits.train_edges.to(self.device)
        train_labels = splits.train_labels.to(self.device)
        val_edges = splits.val_edges.to(self.device)
        val_labels = splits.val_labels.to(self.device)

        for epoch in range(1, self.config.EPOCHS + 1):
            model.train()
            optimizer.zero_grad()

            logits = model(graph_data.x, graph_data.edge_index, train_edges)
            if num_classes > 2:
                loss = criterion(logits, train_labels)
            else:
                loss = criterion(logits.squeeze(), train_labels.float())
            loss.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_logits = model(graph_data.x, graph_data.edge_index, val_edges)
                if num_classes > 2:
                    val_loss = criterion(val_logits, val_labels)
                    val_pred = torch.argmax(val_logits, dim=1)
                    val_metrics = multiclass_metrics(
                        val_labels.cpu().numpy(), val_pred.cpu().numpy()
                    )
                    val_score = val_metrics["f1_weighted"]
                else:
                    val_loss = criterion(val_logits.squeeze(), val_labels.float())
                    val_probs = torch.sigmoid(val_logits).squeeze()
                    val_pred = (val_probs > 0.5).long()
                    val_metrics = binary_metrics(
                        val_labels.cpu().numpy(),
                        val_pred.cpu().numpy(),
                        val_probs.cpu().numpy(),
                    )
                    val_score = val_metrics["f1"]

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(loss.item()),
                    "val_loss": float(val_loss.item()),
                    "val_score": float(val_score),
                }
            )

            if val_score > best_val:
                best_val = val_score
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:03d} | "
                    f"Train Loss: {loss.item():.4f} | "
                    f"Val Loss: {val_loss.item():.4f} | "
                    f"Val Score: {val_score:.4f}"
                )

            if patience_counter >= self.config.PATIENCE:
                print(f"Early stopping at epoch {epoch} (no improvement for {self.config.PATIENCE} epochs)")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        return {
            "model": model,
            "history": history,
            "best_val_score": best_val,
        }

    def evaluate(self, model: torch.nn.Module, graph_data: torch.Tensor, splits, num_classes: int) -> Dict:
        """Evaluate the model on the held-out edges."""
        model.eval()
        graph_data = graph_data.to(self.device)
        test_edges = splits.test_edges.to(self.device)
        test_labels = splits.test_labels.to(self.device)

        with torch.no_grad():
            logits = model(graph_data.x, graph_data.edge_index, test_edges)
            if num_classes > 2:
                preds = torch.argmax(logits, dim=1)
                metrics = multiclass_metrics(
                    test_labels.cpu().numpy(), preds.cpu().numpy()
                )
            else:
                probs = torch.sigmoid(logits).squeeze()
                preds = (probs > 0.5).long()
                metrics = binary_metrics(
                    test_labels.cpu().numpy(),
                    preds.cpu().numpy(),
                    probs.cpu().numpy(),
                )
        return {
            "predictions": preds.cpu().numpy(),
            "metrics": metrics,
        }
