import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from tcrgnn import evaluate_model, load_test_file

MODEL_FILE = "/scratch/project/tcr_ml/gnn_release/research_scripts/soft_label_model/best_model.pt"
t1d_graphs_dir = Path(
    "/scratch/project/tcr_ml/gnn_release/test_data_v2/seekgene/processed"
)
OUT_CSV = Path("seekgene_sample_scores.csv")


def process_control_file(control_file: Path):
    """
    Worker function for a single control file.
    Returns (sources, sequences, scores) for that file.
    """
    control_sample_data = load_test_file(control_file)

    control_sample_scores = np.atleast_1d(
        evaluate_model(MODEL_FILE, control_sample_data)
    )

    # out_dir = Path("results/aml_zero")
    # out_dir.mkdir(exist_ok=True)
    # out_path = out_dir / control_file.with_suffix(".scores.txt").name
    # write_scores_to_txt(control_sample_scores, out_path)

    seqs = [getattr(g, "original_characters", None) for g in control_sample_data]
    n = len(seqs)

    if len(control_sample_scores) != n:
        raise ValueError(
            f"Mismatch: {control_file.name} has {n} sequences "
            f"but {len(control_sample_scores)} scores."
        )

    sources = list(itertools.repeat(control_file.name, n))
    scores = control_sample_scores.astype(np.float32).tolist()  # smaller dtype

    return sources, seqs, scores


if __name__ == "__main__":
    if OUT_CSV.exists():
        OUT_CSV.unlink()  # delete the file
    control_files = sorted(p for p in t1d_graphs_dir.iterdir() if p.is_file())

    # Do NOT use Pool when using CUDA
    for cf in control_files:
        sources, sequences, scores = process_control_file(cf)
        chunk_df = pd.DataFrame(
            {"source": sources, "sequence": sequences, "scores": scores},
            copy=False,
        )
        # append to CSV as before
        mode = "w" if not OUT_CSV.exists() else "a"
        header = not OUT_CSV.exists()
        chunk_df.to_csv(OUT_CSV, mode=mode, index=False, header=header)
