import argparse
import yaml
from pathlib import Path
import sys
from sgi_eval.config import PROJECT_ROOT

# Ensure Python can find our src module
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from src.sgi_eval.dataset_loaders import HaluEvalLoader, MedHalluLoader
from src.sgi_eval.pipeline import EmbeddingGenerator, compute_sgi, SGIThresholdTuner, SGIEvaluator
from src.sgi_eval import config

def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_loader(dataset_name: str):
    """Factory to fetch the correct dataset loader."""
    loaders = {
        "HaluEval": HaluEvalLoader,
        "MedHallu": MedHalluLoader
    }
    if dataset_name not in loaders:
        raise ValueError(f"Dataset {dataset_name} is not supported.")
    return loaders[dataset_name]()

def main():
    parser = argparse.ArgumentParser(description="Run SGI Evaluation Pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--model", type=str, help="Override model key (e.g., mpnet, bge)")
    args = parser.parse_args()

    # Load parameters
    cfg = load_config(PROJECT_ROOT / args.config)
    
    # Overrides
    model_key = args.model if args.model else cfg["model"]["key"]
    dataset_name = cfg["dataset"]["name"]
    n_samples = cfg["dataset"]["n_samples"]
    halluc_prob = cfg["dataset"]["hallucination_prob"]
    thresh_method = cfg["pipeline"]["threshold_method"]

    qrcl_folder_name = config.get_qrcl_name(dataset_name, n_samples, halluc_prob)

    print(f"=== Starting SGI Pipeline: {cfg['experiment_name']} ===")
    
    # 1. Unification
    print(f"\n[1/5] Loading and unifying {dataset_name} data...")
    loader = get_loader(dataset_name)
    qrcl_ds = loader.get_qrcl_dataset(n=n_samples, hallucination_prob=halluc_prob)

    # 2. Embedding
    print(f"\n[2/5] Generating text embeddings using {model_key}...")
    embedder = EmbeddingGenerator(model_key=model_key)
    embedded_ds = embedder.get_embeddings(qrcl_ds, dataset_name, qrcl_folder_name)

    # 3. SGI Calculation
    print("\n[3/5] Computing Semantic Grounding Index...")
    sgi_ds = compute_sgi(embedded_ds)

    # 4. Threshold Tuning
    print("\n[4/5] Tuning hallucination detection thresholds...")
    tuner = SGIThresholdTuner(sgi_ds)
    thresholds = tuner.tune()
    
    plot_path = config.DATA_DIR / f"{cfg['experiment_name']}_diagnostics.png"
    tuner.plot_threshold_diagnostics(save_path=str(plot_path))
    
    final_ds = tuner.apply(method=thresh_method)

    # 5. Evaluation
    print("\n[5/5] Executing final evaluation...")
    evaluator = SGIEvaluator(final_ds)
    evaluator.evaluate_all()

if __name__ == "__main__":
    main()