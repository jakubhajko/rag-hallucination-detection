import random
from datasets import load_dataset, Dataset
from .base import BaseLoader
import config

class HaluEvalLoader(BaseLoader):
    def __init__(self, dataset_name="HaluEval"):
        super().__init__(dataset_name)

    def load_raw(self, file_name="qa_data.json") -> Dataset:
        file_path = self.raw_path / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"HaluEval data not found at {file_path}")
            
        return load_dataset("json", data_files=str(file_path), split="train")

    def transform(self, raw_ds: Dataset, n: int, hallucination_prob: float) -> Dataset:
        sampled_raw = raw_ds.shuffle(seed=42).select(range(n))

        def process_batch(batch):
            res_questions = []
            res_contexts = []
            res_responses = []
            res_labels = []

            for i in range(len(batch['question'])):
                is_hallucinated = random.random() < hallucination_prob
                
                res_questions.append(batch['question'][i])
                res_contexts.append(batch['knowledge'][i])
                
                if is_hallucinated:
                    res_responses.append(batch['hallucinated_answer'][i])
                    res_labels.append(1)
                else:
                    res_responses.append(batch['right_answer'][i])
                    res_labels.append(0)

            return {
                config.COL_QUESTION: res_questions,
                config.COL_CONTEXT: res_contexts,
                config.COL_RESPONSE: res_responses,
                config.COL_LABEL: res_labels
            }

        qrcl_ds = sampled_raw.map(
            process_batch, 
            batched=True, 
            remove_columns=raw_ds.column_names
        )
        
        return qrcl_ds
    
# loader = HaluEvalLoader("HaluEval")
# # Get 5000 samples, with 70% hallucinations
# dataset = loader.get_dataset(n=5000, hallucination_prob=0.7)

# print(type(dataset)) 
# # Output: {'question': '...', 'context': '...', 'response': '...', 'label': 1}