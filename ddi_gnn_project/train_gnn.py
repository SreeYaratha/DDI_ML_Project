"""
Entry point for training a GNN-based DDI predictor.
"""
import os
import random

import numpy as np
import torch

from config import config
from data.graph_builder import DDIGraphBuilder
from data.splitter import ColdStartSplitter
from models.trainer import GNNTrainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(config.SEED)
    print("=" * 60)
    print("GNN DDI TRAINING")
    print("=" * 60)

    builder = DDIGraphBuilder(
        ddi_path=config.DDI_PATH,
        drug_features_path=config.DRUG_FEATURES_PATH,
        drug1_col=config.DRUG1_COL,
        drug2_col=config.DRUG2_COL,
        interaction_col=config.INTERACTION_COL,
        min_interactions=config.MIN_INTERACTIONS,
        binary_labels=config.BINARY_LINK_PREDICTION,
        max_classes=config.MAX_INTERACTION_CLASSES,
    )

    graph_data = builder.load_and_build()
    stats = builder.get_statistics()

    print("\nGraph statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    if config.BINARY_LINK_PREDICTION:
        num_classes = 2
        print("\nInteraction classes: binary (link prediction)")
        model_type = "link"
    else:
        num_classes = len(builder.label_encoder.classes_)
        print(f"\nInteraction classes: {num_classes}")
        model_type = config.MODEL_TYPE

    splitter = ColdStartSplitter(
        test_ratio=config.TEST_RATIO,
        val_ratio=config.VAL_RATIO,
        seed=config.SEED,
    )
    splits = splitter.split(graph_data, cold_start=config.COLD_START)

    print("\nEdge splits:")
    print(f"  Train edges: {len(splits.train_edges)}")
    print(f"  Val edges:   {len(splits.val_edges)}")
    print(f"  Test edges:  {len(splits.test_edges)}")

    trainer = GNNTrainer(config)
    # Force link model when binary to avoid accidental mismatch
    if config.BINARY_LINK_PREDICTION:
        config.MODEL_TYPE = "link"
    else:
        config.MODEL_TYPE = model_type

    results = trainer.train(graph_data, splits, num_classes=num_classes)

    eval_results = trainer.evaluate(results["model"], graph_data, splits, num_classes=num_classes)
    print("\nEvaluation metrics:")
    for k, v in eval_results["metrics"].items():
        print(f"  {k}: {v:.4f}")

    os.makedirs("ddi_gnn_project/artifacts", exist_ok=True)
    torch.save(results["model"].state_dict(), "ddi_gnn_project/artifacts/best_gnn_model.pth")
    print("\nSaved best model to ddi_gnn_project/artifacts/best_gnn_model.pth")

    print("\n" + "=" * 60)
    print("GNN training complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
