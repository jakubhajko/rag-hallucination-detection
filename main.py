from src.dataset_loaders.halueval_loader import HaluEvalLoader
from src.pipeline.embeddings_generator import EmbeddingGenerator
from src.pipeline.sgi import compute_sgi
from src.pipeline.threshold_tuner import SGIThresholdTuner
from src.pipeline.evaluator import SGIEvaluator
import config

def main():
    # --- Parameters ---
    N_SAMPLES = 5000
    HALLUC_PROB = 0.5
    MODEL_KEY = "bge"
    QRCL_FOLDER_NAME = config.get_qrcl_name("HaluEval", N_SAMPLES, HALLUC_PROB)
    
    print("=== Step 1: Loading & Unifying Data ===")
    loader = HaluEvalLoader()
    qrcl_ds = loader.get_qrcl_dataset(n=N_SAMPLES, hallucination_prob=HALLUC_PROB)

    print("\n=== Step 2: Generating Embeddings ===")
    embedder = EmbeddingGenerator(model_key=MODEL_KEY)
    embedded_ds = embedder.get_embeddings(
        qrcl_ds, 
        dataset_name="HaluEval", 
        qrcl_folder_name=QRCL_FOLDER_NAME
    )

    print("\n=== Step 3: Computing SGI Scores ===")
    sgi_ds = compute_sgi(embedded_ds)

    print("\n=== Step 4: Tuning Thresholds ===")
    tuner = SGIThresholdTuner(sgi_ds)
    thresholds = tuner.tune()  # Calculates both Youden and F1
    
    # Save the plot to the data directory
    plot_path = str(config.PROJECT_ROOT / "data" / "tuning_diagnostics.png")
    tuner.plot_threshold_diagnostics(save_path=plot_path)
    
    # You can easily swap "youden" for "f1" here based on what you see in the plot!
    final_ds = tuner.apply(method="youden")

    print("\n=== Step 5: Final Evaluation ===")
    evaluator = SGIEvaluator(final_ds)
    evaluator.evaluate_all()

if __name__ == "__main__":
    main()