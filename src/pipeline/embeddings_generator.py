import sys
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from datasets import load_from_disk

# Add project root to sys.path
root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))
import config

class EmbeddingGenerator:
    MODELS = {
        "mpnet": "all-mpnet-base-v2",   # 384 tokens
        "minilm": "all-MiniLM-L6-v2",   # 256 tokens
        "bge": "BAAI/bge-base-en-v1.5", # 512 tokens
        "e5": "intfloat/e5-base-v2",    # 512 tokens
        "gte": "thenlper/gte-base"      # 512 tokens
    }

    def __init__(self, model_key="mpnet"):
        if model_key not in self.MODELS:
            raise ValueError(f"Model {model_key} not supported.")
        self.model_key = model_key
        self.model = SentenceTransformer(self.MODELS[model_key], device=config.DEVICE)

    def get_embeddings(self, qrcl_ds, dataset_name: str, qrcl_folder_name: str):
        """
        qrcl_folder_name: the name of the folder from the previous step (e.g., 'HaluEval_n5000_p0.5')
        """
        # Append the model name to the QRCL folder name
        folder_name = f"{qrcl_folder_name}_{self.model_key}"
        save_path = config.EMBEDDED_DATA_DIR / dataset_name / folder_name

        if save_path.exists():
            print(f"Loading cached embeddings from: {save_path}")
            return load_from_disk(str(save_path))

        print(f"Generating embeddings using {self.model_key} on {config.DEVICE}...")
        
        # Ensure the target directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        def embed_batch(batch):
            # Normalize=True is required for the SGI math [cite: 120, 121]
            q_emb = self.model.encode(batch[config.COL_QUESTION], normalize_embeddings=True)
            c_emb = self.model.encode(batch[config.COL_CONTEXT], normalize_embeddings=True)
            r_emb = self.model.encode(batch[config.COL_RESPONSE], normalize_embeddings=True)
            return {"q_emb": q_emb.tolist(), "c_emb": c_emb.tolist(), "r_emb": r_emb.tolist()}

        embedded_ds = qrcl_ds.map(embed_batch, batched=True, batch_size=32)
        embedded_ds.save_to_disk(str(save_path))
        return embedded_ds