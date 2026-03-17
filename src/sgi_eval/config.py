import os
from pathlib import Path
import torch

# Intention: Centralize project paths to ensure data is strictly sandboxed.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
QRCL_DATA_DIR = DATA_DIR / "qrcl"
EMBEDDED_DATA_DIR = DATA_DIR / "embedded"

# Intention: Define the strict unified schema required by the pipeline.
COL_QUESTION = "question"
COL_CONTEXT = "context"
COL_RESPONSE = "response"
COL_LABEL = "label"

# Intention: Auto-detect the best available hardware accelerator.
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def get_qrcl_name(dataset_name: str, n_samples: int, halluc_prob: float) -> str:
    """Standardized naming convention to safely cache and retrieve experimental data."""
    n_str = f"n{n_samples}" if n_samples else "all"
    return f"{dataset_name}_{n_str}_p{halluc_prob}"