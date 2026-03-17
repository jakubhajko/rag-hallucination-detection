import random
import urllib.request
from datasets import load_dataset, Dataset
from .base import BaseLoader
from .. import config

class HaluEvalLoader(BaseLoader):
    """Intention: Adapt the adversarially generated HaluEval dataset into our pipeline."""
    
    def __init__(self, dataset_name="HaluEval"):
        super().__init__(dataset_name)
        # The official raw data link for automatic downloading
        self.data_url = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/refs/heads/main/data/qa_data.json"

    def load_raw(self, file_name="qa_data.json") -> Dataset:
        """Intention: Fetch raw data from disk, or download it from the official repo if missing."""
        file_path = self.raw_path / file_name
        
        # Auto-download mechanism
        if not file_path.exists():
            print(f"Downloading HaluEval raw data from {self.data_url}...")
            # Ensure the raw path directory exists before downloading
            self.raw_path.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(self.data_url, str(file_path))
            print("Download complete!")

        return load_dataset("json", data_files=str(file_path), split="train")

    def transform(self, raw_ds: Dataset, n: int, hallucination_prob: float, **kwargs) -> Dataset:
        """Intention: Sample and map to the unified QRCL schema, applying the requested hallucination ratio."""
        # Ensure we don't try to sample more data than the dataset actually contains
        sampled_raw = raw_ds.shuffle(seed=42).select(range(min(n, len(raw_ds)))) if n else raw_ds

        def process_batch(batch):
            res = {config.COL_QUESTION: [], config.COL_CONTEXT: [], config.COL_RESPONSE: [], config.COL_LABEL: []}

            for i in range(len(batch['question'])):
                is_hallucinated = random.random() < hallucination_prob
                
                res[config.COL_QUESTION].append(batch['question'][i])
                res[config.COL_CONTEXT].append(batch['knowledge'][i])
                res[config.COL_LABEL].append(1 if is_hallucinated else 0)
                
                answer_key = 'hallucinated_answer' if is_hallucinated else 'right_answer'
                res[config.COL_RESPONSE].append(batch[answer_key][i])

            return res

        return sampled_raw.map(process_batch, batched=True, remove_columns=raw_ds.column_names)