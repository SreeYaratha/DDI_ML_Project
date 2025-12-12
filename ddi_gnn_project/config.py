"""
Configuration for Graph Neural Network DDI prediction.
"""


class GNNConfig:
    # Paths (defaults align with existing processed artifacts)
    DDI_PATH = "data/raw/drug_ddi.csv"
    DRUG_FEATURES_PATH = "results/drug_features.csv"

    # Column names in the raw DDI file
    DRUG1_COL = "Drug 1"
    DRUG2_COL = "Drug 2"
    INTERACTION_COL = "Interaction Description"

    # Model selection
    MODEL_TYPE = "edgegnn"  # options: edgegnn, gat, link (overridden if binary)

    # Architecture
    HIDDEN_DIM = 256
    NUM_LAYERS = 3
    HEADS = 4  # only used for GAT
    DROPOUT = 0.3

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    EPOCHS = 50
    PATIENCE = 10

    # Data split
    TEST_RATIO = 0.2
    VAL_RATIO = 0.1
    SEED = 42

    # Cold-start settings
    COLD_START = True  # split by nodes to simulate unseen drugs
    MIN_INTERACTIONS = 3  # minimum degree to keep a node

    # Feature handling
    USE_PRETRAINED_EMBEDDINGS = False
    EMBEDDING_DIM = 128  # used if USE_PRETRAINED_EMBEDDINGS is True

    # Label handling
    BINARY_LINK_PREDICTION = False  # set True to collapse interactions to binary (any interaction vs none)
    MAX_INTERACTION_CLASSES = 64  # cap on distinct interaction labels when binary is False


config = GNNConfig()
