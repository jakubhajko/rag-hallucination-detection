import torch
from datasets import Dataset

def calculate_angular_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Computes geodesic distance on the unit hypersphere[cite: 41, 42]."""
    # The paper uses clip to prevent numerical errors from domain violations [cite: 121]
    cos_sim = torch.sum(a * b, dim=-1)
    return torch.acos(torch.clamp(cos_sim, -1.0, 1.0))

def compute_sgi(embedded_ds: Dataset, epsilon: float = 1e-8) -> Dataset:
    """
    Accepts the embedded HF Dataset, computes theta and SGI, 
    and returns a new Dataset with the scores attached.
    """
    def batch_sgi(batch):
        # Convert the lists of floats back into PyTorch tensors for vectorized math
        q = torch.tensor(batch["q_emb"])
        c = torch.tensor(batch["c_emb"])
        r = torch.tensor(batch["r_emb"])

        theta_rq = calculate_angular_distance(r, q)
        theta_rc = calculate_angular_distance(r, c)
        theta_qc = calculate_angular_distance(q, c)  # Optional: could be used for analysis 
        
        # SGI Formula: theta(r,q) / theta(r,c) [cite: 48, 49]
        # Epsilon is added to the denominator to prevent division by zero [cite: 122]
        sgi_scores = theta_rq / (theta_rc + epsilon)
        
        return {
            "theta_qc": theta_qc.tolist(),  # Optional: for analysis, not used in SGI calculation
            "theta_rq": theta_rq.tolist(),
            "theta_rc": theta_rc.tolist(),
            "sgi_score": sgi_scores.tolist()
        }

    print("Computing Semantic Grounding Index scores...")
    # Batched map applies the math efficiently over the whole dataset
    return embedded_ds.map(batch_sgi, batched=True)