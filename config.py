from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent

# Directory Structure
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw_format"
QRCL_DATA_DIR = PROJECT_ROOT / "data" / "qrcl_format"
EMBEDDED_DATA_DIR = PROJECT_ROOT / "data" / "embedded_format"

# Column Constants
COL_QUESTION = "question"
COL_CONTEXT = "context"
COL_RESPONSE = "response"
COL_LABEL = "label"

# Device Support (CUDA, MPS for Mac, or CPU)
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# Embedding Model Context Windows 
# all-mpnet-base-v2: 384 tokens
# all-MiniLM-L6-v2: 256 tokens
# bge-base-en-v1.5: 512 tokens
# e5-base-v2: 512 tokens
# gte-base: 512 tokens

def get_qrcl_name(dataset_name, n, p):
    return f"{dataset_name}_n{n}_p{p}"

def get_embedded_name(dataset_name, n, p, model_name):
    return f"{dataset_name}_n{n}_p{p}_{model_name}"

# Ensure directories exist
for path in [RAW_DATA_DIR, QRCL_DATA_DIR, EMBEDDED_DATA_DIR]:
    path.mkdir(parents=True, exist_ok=True)