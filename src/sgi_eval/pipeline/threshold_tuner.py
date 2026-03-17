import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, f1_score
from datasets import Dataset
from .. import config

class SGIThresholdTuner:
    """Intention: Identify optimal SGI cutoffs for binary classification (Valid vs. Hallucinated)."""
    
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.labels = np.array(dataset[config.COL_LABEL])
        self.scores = np.array(dataset["sgi_score"])
        
        self.opt_youden = None
        self.opt_f1 = None

    def tune(self) -> dict:
        """Intention: Calculate optimal thresholds using both statistical (Youden) and practical (F1) metrics."""
        # ROC relies on higher scores predicting positive class (1 = hallucination). 
        # Since lower SGI indicates hallucination[cite: 53], we invert the scores.
        inverted_scores = -self.scores
        self.fpr, self.tpr, self.roc_thresholds = roc_curve(self.labels, inverted_scores)
        
        # Youden's J Statistic
        optimal_idx = np.argmax(self.tpr - self.fpr)
        self.opt_youden = -self.roc_thresholds[optimal_idx]
        
        # Max F1 Score search
        best_f1, optimal_inverted_f1 = 0, 0
        for thresh in self.roc_thresholds[1:]:
            preds = (inverted_scores >= thresh).astype(int)
            current_f1 = f1_score(self.labels, preds)
            if current_f1 > best_f1:
                best_f1 = current_f1
                optimal_inverted_f1 = thresh
                
        self.opt_f1 = -optimal_inverted_f1
        return {"youden": self.opt_youden, "f1": self.opt_f1}

    def apply(self, method: str = "youden") -> Dataset:
        """Intention: Append discrete classification labels to the dataset based on the chosen threshold."""
        if not self.opt_youden:
            self.tune()

        chosen_thresh = self.opt_youden if method == "youden" else self.opt_f1
        print(f"Applying '{method}' threshold ({chosen_thresh:.4f})...")

        def apply_threshold_batch(batch):
            preds = [1 if s < chosen_thresh else 0 for s in batch["sgi_score"]]
            is_correct = [1 if p == l else 0 for p, l in zip(preds, batch[config.COL_LABEL])]
            return {"predicted_label": preds, "is_correct": is_correct}

        return self.dataset.map(apply_threshold_batch, batched=True)

    def plot_threshold_diagnostics(self, save_path: str = None):
        """Intention: Provide visual validation of how well the threshold separates the score distributions."""
        if self.opt_youden is None:
            self.tune()

        valid_scores = self.scores[self.labels == 0]
        halluc_scores = self.scores[self.labels == 1]

        fig, ax = plt.subplots(figsize=(8, 5))
        bins = np.linspace(min(self.scores), max(self.scores), 50)
        
        ax.hist(valid_scores, bins=bins, alpha=0.5, color='green', density=True, label='Valid (0)')
        ax.hist(halluc_scores, bins=bins, alpha=0.5, color='red', density=True, label='Hallucination (1)')
        
        ax.axvline(self.opt_youden, color='black', linestyle='--', lw=2, label=f'Youden ({self.opt_youden:.2f})')
        ax.axvline(self.opt_f1, color='purple', linestyle=':', lw=2, label=f'F1 ({self.opt_f1:.2f})')
        
        ax.set_title('SGI Score Distributions')
        ax.set_xlabel('Semantic Grounding Index (SGI)')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()