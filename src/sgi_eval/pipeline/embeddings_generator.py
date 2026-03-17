import torch
from sentence_transformers import SentenceTransformer
from datasets import Dataset, load_from_disk
from .. import config

class EmbeddingGenerator:
    """Intention: Map strings to semantic vector spaces using state-of-the-art sentence transformers."""
    
    MODELS = {
        "mpnet": "all-mpnet-base-v2",
        "minilm": "all-MiniLM-L6-v2",
        "bge": "BAAI/bge-base-en-v1.5",
        "e5": "intfloat/e5-base-v2",
        "gte": "thenlper/gte-base"
    }

    def __init__(self, model_key="bge"):
        if model_key not in self.MODELS:
            raise ValueError(f"Model {model_key} not supported.")
        self.model_key = model_key
        self.model = SentenceTransformer(self.MODELS[model_key], device=config.DEVICE)

    def get_embeddings(self, qrcl_ds: Dataset, dataset_name: str, qrcl_folder_name: str) -> Dataset:
        """Intention: Generate and cache L2-normalized embeddings required for spherical geometry."""
        folder_name = f"{qrcl_folder_name}_{self.model_key}"
        save_path = config.EMBEDDED_DATA_DIR / dataset_name / folder_name

        if save_path.exists():
            print(f"Loading cached embeddings from: {save_path}")
            return load_from_disk(str(save_path))

        print(f"Generating embeddings using {self.model_key} on {config.DEVICE}...")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        def embed_batch(batch):
            # L2 normalization is strictly required to locate embeddings on the unit hypersphere[cite: 36, 120].
            return {
                "q_emb": self.model.encode(batch[config.COL_QUESTION], normalize_embeddings=True).tolist(),
                "c_emb": self.model.encode(batch[config.COL_CONTEXT], normalize_embeddings=True).tolist(),
                "r_emb": self.model.encode(batch[config.COL_RESPONSE], normalize_embeddings=True).tolist()
            }

        embedded_ds = qrcl_ds.map(embed_batch, batched=True, batch_size=32)
        embedded_ds.save_to_disk(str(save_path))
        return embedded_ds