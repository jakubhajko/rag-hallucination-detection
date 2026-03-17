import random
from datasets import load_dataset, Dataset, concatenate_datasets
from .base import BaseLoader
from .. import config

class MedHalluLoader(BaseLoader):
    """
    Intention: Load and unify the MedHallu dataset. 
    Merges human-labeled and artificial splits and allows filtering by difficulty.
    """
    
    def __init__(self, dataset_name="MedHallu"):
        super().__init__(dataset_name)

    def load_raw(self) -> Dataset:
        """Intention: Fetch both splits from the Hugging Face hub and merge them to a 10k dataset."""
        print("Downloading 'pqa_labeled' split...")
        labeled_ds = load_dataset(
            "UTAustin-AIHealth/MedHallu", "pqa_labeled", 
            split="train", cache_dir=str(self.raw_path)
        )
        
        print("Downloading 'pqa_artificial' split...")
        artificial_ds = load_dataset(
            "UTAustin-AIHealth/MedHallu", "pqa_artificial", 
            split="train", cache_dir=str(self.raw_path)
        )
        
        print("Merging splits into a single 10,000 sample dataset...")
        return concatenate_datasets([labeled_ds, artificial_ds])

    def transform(self, raw_ds: Dataset, n: int, hallucination_prob: float, difficulty: str = "all", **kwargs) -> Dataset:
        """
        Intention: Filter by difficulty, sample, and map to the unified QRCL schema.
        `difficulty` options: "all", "easy", "medium", "hard".
        """
        if difficulty.lower() != "all":
            print(f"Filtering dataset for difficulty level: {difficulty}")
            raw_ds = raw_ds.filter(lambda x: x['Difficulty Level'].lower() == difficulty.lower())

            max_available = len(raw_ds)
            if n and n > max_available:
                print(f"Warning: Only {max_available} samples available for difficulty '{difficulty}'. Adjusting 'n'.")
                n = max_available

        sampled_raw = raw_ds.shuffle(seed=42).select(range(n)) if n else raw_ds

        def process_batch(batch):
            res = {config.COL_QUESTION: [], config.COL_CONTEXT: [], config.COL_RESPONSE: [], config.COL_LABEL: []}

            # Helper to guarantee the output is a single, clean string
            def _safe_string(val):
                if val is None:
                    return ""
                if isinstance(val, list):
                    return " ".join([str(v) for v in val])
                return str(val)

            for i in range(len(batch['Question'])):
                is_hallucinated = random.random() < hallucination_prob
                
                # Sanitize inputs before appending
                res[config.COL_QUESTION].append(_safe_string(batch['Question'][i]))
                res[config.COL_CONTEXT].append(_safe_string(batch['Knowledge'][i]))
                res[config.COL_LABEL].append(1 if is_hallucinated else 0)
                
                answer_key = 'Hallucinated Answer' if is_hallucinated else 'Ground Truth'
                res[config.COL_RESPONSE].append(_safe_string(batch[answer_key][i]))

            return res

        return sampled_raw.map(process_batch, batched=True, remove_columns=raw_ds.column_names)