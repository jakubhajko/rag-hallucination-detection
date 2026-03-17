import torch
from datasets import Dataset
from .. import config

def compute_sgi(embedded_ds: Dataset, epsilon: float = 1e-8) -> Dataset:
    """
    Intention: Vectorize the Semantic Grounding Index math across batches, 
    executing on the fastest available hardware accelerator before returning to standard CPU lists.
    """
    device = config.DEVICE
    print(f"Computing SGI scores on {device}...")

    def batch_sgi(batch):
        # Move lists to tensors and explicitly place on the compute device
        q = torch.tensor(batch["q_emb"], device=device)
        c = torch.tensor(batch["c_emb"], device=device)
        r = torch.tensor(batch["r_emb"], device=device)

        # Cosine similarity clamped to prevent domain violations during arccos.
        cos_rq = torch.clamp(torch.sum(r * q, dim=-1), -1.0, 1.0)
        cos_rc = torch.clamp(torch.sum(r * c, dim=-1), -1.0, 1.0)

        # Compute geodesic distance on the unit hypersphere[cite: 41, 42].
        theta_rq = torch.acos(cos_rq)
        theta_rc = torch.acos(cos_rc)

        # Calculate SGI. Epsilon prevents division by zero if context and response overlap perfectly[cite: 49, 122].
        sgi_scores = theta_rq / (theta_rc + epsilon)

        # Return to CPU memory to comply with Hugging Face Dataset storage requirements
        return {
            "theta_rq": theta_rq.cpu().tolist(),
            "theta_rc": theta_rc.cpu().tolist(),
            "sgi_score": sgi_scores.cpu().tolist()
        }

    return embedded_ds.map(batch_sgi, batched=True)