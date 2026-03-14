import os
from abc import ABC, abstractmethod
from pathlib import Path
from datasets import Dataset, load_from_disk
import sys
# Add project root to sys.path so we can import config
root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))
import config

class BaseLoader(ABC):
    """
    Abstract Base Class for Dataset Unification Layer.
    Enforces the QRCL (Question, Response, Context, Label) schema.
    """
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.raw_path = config.RAW_DATA_DIR / dataset_name
        self.processed_path = config.QRCL_DATA_DIR / dataset_name

        # Ensure directories exist
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def load_raw(self, file_name: str) -> Dataset:
        """Load the raw files into a Hugging Face Dataset."""
        pass

    @abstractmethod
    def transform(self, raw_ds: Dataset, n: int, hallucination_prob: float) -> Dataset:
        """Transform raw data into QRCL format with specific sampling logic."""
        pass

    def get_qrcl_dataset(self, n: int = None, hallucination_prob: float = 0.5) -> Dataset:
        """Handles caching and retrieval of the QRCL unified text format."""
        folder_name = config.get_qrcl_name(self.dataset_name, n, hallucination_prob)
        save_path = config.QRCL_DATA_DIR / self.dataset_name / folder_name

        if save_path.exists():
            print(f"Loading cached QRCL data from: {save_path}")
            return load_from_disk(str(save_path))

        print(f"Generating QRCL data for {self.dataset_name}...")
        raw_ds = self.load_raw()
        qrcl_ds = self.transform(raw_ds, n, hallucination_prob)
        
        qrcl_ds.save_to_disk(str(save_path))
        return qrcl_ds