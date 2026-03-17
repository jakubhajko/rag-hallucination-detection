import numpy as np
from scipy.stats import ttest_ind
from sklearn.metrics import roc_auc_score
from datasets import Dataset
from .. import config

class SGIEvaluator:
    """Intention: Compute standard paper metrics to validate the geometric bounding hypothesis."""
    
    def __init__(self, dataset: Dataset):
        self.labels = np.array(dataset[config.COL_LABEL])
        self.scores = np.array(dataset["sgi_score"])
        
        self.valid_scores = self.scores[self.labels == 0]
        self.halluc_scores = self.scores[self.labels == 1]

    def _cohens_d(self) -> float:
        """Intention: Measure effect size using pooled standard deviation[cite: 127]."""
        n1, n2 = len(self.valid_scores), len(self.halluc_scores)
        var1, var2 = np.var(self.valid_scores, ddof=1), np.var(self.halluc_scores, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        return (np.mean(self.valid_scores) - np.mean(self.halluc_scores)) / pooled_std

    def evaluate_all(self) -> dict:
        """Intention: Output all summary statistics for the experiment."""
        # Welch's t-test handles unequal variances natively[cite: 127].
        _, p_val = ttest_ind(self.valid_scores, self.halluc_scores, equal_var=False)
        auc = roc_auc_score(self.labels, -self.scores)
        d_score = self._cohens_d()

        metrics = {
            "mean_sgi_valid": np.mean(self.valid_scores),
            "mean_sgi_halluc": np.mean(self.halluc_scores),
            "cohens_d": d_score,
            "roc_auc": auc,
            "p_value": p_val
        }
        
        print("\n=== Final SGI Evaluation Metrics ===")
        print(f"Mean SGI (Valid):       {metrics['mean_sgi_valid']:.3f}")
        print(f"Mean SGI (Halluc):      {metrics['mean_sgi_halluc']:.3f}")
        print(f"Effect Size (Cohen's d): +{metrics['cohens_d']:.2f}")
        print(f"ROC-AUC:                {metrics['roc_auc']:.3f}")
        print(f"p-value (Welch's t):    {metrics['p_value']:.2e}\n")
        
        return metrics