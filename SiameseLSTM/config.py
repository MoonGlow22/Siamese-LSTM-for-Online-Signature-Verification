import torch
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

DATA_DIRECTORY = os.path.join(parent_dir, "Signatures")
OUTPUT_DIRECTORY = os.path.join(parent_dir, "training_outputs")

# Model parameters
MODEL_CONFIG = {
    'input_size': 3,
    'hidden_size': 128,
    'dropout_rate': 0.3,
    'num_layers': 2,
    'use_attention': False,
    "dense_size": 128,
    "dense_dropout": 0.2
}

# Training parameters
TRAINING_CONFIG = {
    'batch_size': 32,
    'num_epochs': 30,
    'learning_rate': 0.001,
    'margin': 1.5,
    'patience': 15,
    'max_pairs_per_person': 200
}

