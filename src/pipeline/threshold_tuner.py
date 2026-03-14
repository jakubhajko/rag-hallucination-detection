import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, f1_score
from datasets import Dataset
import sys
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))
import config

class SGIThresholdTuner:
    def __init__(self, dataset: Dataset):
        """Expects a Dataset with 'sgi_score' and 'label' columns."""
        self.dataset = dataset
        self.labels = np.array(dataset[config.COL_LABEL])
        self.scores = np.array(dataset["sgi_score"])
        
        self.optimal_threshold_youden = None
        self.optimal_threshold_f1 = None
        self.fpr = None
        self.tpr = None
        self.roc_thresholds = None

    def tune(self) -> dict:
        """Finds optimal SGI thresholds using both Youden's J and F1 Score."""
        # Invert scores because standard ROC expects higher scores to predict class 1 (Hallucination)
        inverted_scores = -self.scores
        self.fpr, self.tpr, self.roc_thresholds = roc_curve(self.labels, inverted_scores)
        
        # --- Method 1: Youden's J Statistic ---
        youden_j = self.tpr - self.fpr
        optimal_idx_youden = np.argmax(youden_j)
        self.optimal_threshold_youden = -self.roc_thresholds[optimal_idx_youden]
        
        # --- Method 2: Maximum F1 Score ---
        best_f1 = 0
        optimal_inverted_f1 = 0
        
        for thresh in self.roc_thresholds[1:]:
            preds = (inverted_scores >= thresh).astype(int)
            current_f1 = f1_score(self.labels, preds)
            if current_f1 > best_f1:
                best_f1 = current_f1
                optimal_inverted_f1 = thresh
                
        self.optimal_threshold_f1 = -optimal_inverted_f1
        
        print(f"Optimal SGI Threshold (Youden's J): {self.optimal_threshold_youden:.4f}")
        print(f"Optimal SGI Threshold (F1 Score):   {self.optimal_threshold_f1:.4f}")
        
        return {
            "youden": self.optimal_threshold_youden,
            "f1": self.optimal_threshold_f1
        }

    def plot_threshold_diagnostics(self, save_path: str = None):
        """Visualizes both optimal thresholds on the ROC and Density plots."""
        if self.optimal_threshold_youden is None:
            self.tune()

        valid_scores = self.scores[self.labels == 0]
        halluc_scores = self.scores[self.labels == 1]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # --- Plot 1: ROC Curve ---
        ax1.plot(self.fpr, self.tpr, color='blue', lw=2, label='ROC curve')
        ax1.plot([0, 1], [0, 1], color='gray', linestyle='--')
        
        # Mark Youden on ROC
        optimal_idx_youden = np.argmax(self.tpr - self.fpr)
        ax1.scatter(self.fpr[optimal_idx_youden], self.tpr[optimal_idx_youden], 
                    color='red', s=100, zorder=5, label='Optimal (Youden)')
        
        ax1.set_title('ROC Curve for Hallucination Detection')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.legend(loc="lower right")
        ax1.grid(alpha=0.3)

        # --- Plot 2: Score Distributions ---
        bins = np.linspace(min(self.scores), max(self.scores), 50)
        ax2.hist(valid_scores, bins=bins, alpha=0.5, color='green', density=True, label='Valid (Label=0)')
        ax2.hist(halluc_scores, bins=bins, alpha=0.5, color='red', density=True, label='Hallucination (Label=1)')
        
        # Draw BOTH threshold lines
        ax2.axvline(self.optimal_threshold_youden, color='black', linestyle='dashed', linewidth=2, 
                    label=f'Youden ({self.optimal_threshold_youden:.2f})')
        ax2.axvline(self.optimal_threshold_f1, color='purple', linestyle='dotted', linewidth=2, 
                    label=f'F1 ({self.optimal_threshold_f1:.2f})')
        
        ax2.set_title('SGI Score Distributions')
        ax2.set_xlabel('SGI Score')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Visualization saved to {save_path}")
        plt.show()

    def apply(self, method: str = "youden") -> Dataset:
        """Applies the chosen threshold ('youden' or 'f1') to generate predictions."""
        if self.optimal_threshold_youden is None:
            self.tune()

        chosen_threshold = self.optimal_threshold_youden if method == "youden" else self.optimal_threshold_f1
        print(f"Applying '{method}' threshold ({chosen_threshold:.4f}) to generate predictions...")

        def apply_threshold_batch(batch):
            predictions = []
            is_correct = []
            
            for sgi, label in zip(batch["sgi_score"], batch[config.COL_LABEL]):
                pred = 1 if sgi < chosen_threshold else 0
                predictions.append(pred)
                is_correct.append(1 if pred == label else 0)
                
            return {"predicted_label": predictions, "is_correct": is_correct}

        return self.dataset.map(apply_threshold_batch, batched=True)