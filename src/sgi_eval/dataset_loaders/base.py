from abc import ABC, abstractmethod
from pathlib import Path
from datasets import Dataset, load_from_disk
from .. import config

class BaseLoader(ABC):
    """
    Intention: Provide an abstract contract that all raw datasets must fulfill 
    to enter the pipeline. Guarantees the unified QRCL format.
    """
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.raw_path = config.RAW_DATA_DIR / dataset_name
        self.processed_path = config.QRCL_DATA_DIR / dataset_name

        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def load_raw(self) -> Dataset:
        """Fetches raw data from disk or external sources."""
        pass

    @abstractmethod
    def transform(self, raw_ds: Dataset, n: int, hallucination_prob: float) -> Dataset:
        """Maps bespoke dataset schemas into the unified QRCL schema."""
        pass

    def get_qrcl_dataset(self, n: int = None, hallucination_prob: float = 0.5) -> Dataset:
        """Intention: Manage caching to prevent redundant data processing on repeat runs."""
        folder_name = config.get_qrcl_name(self.dataset_name, n, hallucination_prob)
        save_path = self.processed_path / folder_name

        if save_path.exists():
            print(f"Loading cached QRCL data from: {save_path}")
            return load_from_disk(str(save_path))

        print(f"Generating unified QRCL data for {self.dataset_name}...")
        raw_ds = self.load_raw()
        qrcl_ds = self.transform(raw_ds, n, hallucination_prob)
        
        qrcl_ds.save_to_disk(str(save_path))
        return qrcl_ds