import numpy as np
from scipy.stats import ttest_ind
from sklearn.metrics import roc_auc_score
from datasets import Dataset
import sys
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))
import config

class SGIEvaluator:
    def __init__(self, dataset: Dataset):
        """
        Initializes the evaluator with a dataset containing 'sgi_score' and 'label' columns.
        """
        self.dataset = dataset
        self.labels = np.array(dataset[config.COL_LABEL])
        self.scores = np.array(dataset["sgi_score"])
        
        # Split scores based on our HaluEval loader: 0 = Valid, 1 = Hallucinated
        self.valid_scores = self.scores[self.labels == 0]
        self.halluc_scores = self.scores[self.labels == 1]

    def compute_cohens_d(self) -> float:
        """Computes Cohen's d effect size using pooled standard deviation[cite: 127]."""
        n1, n2 = len(self.valid_scores), len(self.halluc_scores)
        var1 = np.var(self.valid_scores, ddof=1)
        var2 = np.var(self.halluc_scores, ddof=1)
        
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        # Positive d means valid scores are higher than hallucinated scores
        return (np.mean(self.valid_scores) - np.mean(self.halluc_scores)) / pooled_std

    def compute_p_value(self) -> float:
        """Computes statistical significance using Welch's t-test (unequal variances)[cite: 127]."""
        # equal_var=False triggers Welch's t-test in scipy
        stat, p_val = ttest_ind(self.valid_scores, self.halluc_scores, equal_var=False)
        return p_val

    def compute_roc_auc(self) -> float:
        """
        Computes ROC-AUC. 
        Since high SGI predicts Valid (label=0)[cite: 52, 53], we invert the scores 
        so that high (-SGI) predicts Hallucination (label=1) to use standard AUC logic.
        """
        # Inverting scores so higher value = higher chance of hallucination
        inverted_scores = -self.scores
        return roc_auc_score(self.labels, inverted_scores)

    def evaluate_all(self) -> dict:
        """Returns a dictionary of all metrics used in the paper."""
        metrics = {
            "mean_sgi_valid": np.mean(self.valid_scores),
            "mean_sgi_halluc": np.mean(self.halluc_scores),
            "cohens_d": self.compute_cohens_d(),
            "p_value": self.compute_p_value(),
            "roc_auc": self.compute_roc_auc()
        }
        
        # Pretty print results
        print("\n=== SGI Evaluation Results ===")
        print(f"Mean SGI (Valid):       {metrics['mean_sgi_valid']:.3f}")
        print(f"Mean SGI (Halluc):      {metrics['mean_sgi_halluc']:.3f}")
        print(f"Effect Size (Cohen's d): +{metrics['cohens_d']:.2f}")
        print(f"ROC-AUC:                {metrics['roc_auc']:.3f}")
        # Using scientific notation for highly significant p-values
        print(f"p-value (Welch's t):    {metrics['p_value']:.2e}\n")
        
        return metrics