import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from tcrgnn import evaluate_model, load_test_file


def process_control_file(control_file: Path, model_file: Path):
    """
    Worker function for a single control file.
    Returns (sources, sequences, scores) for that file.
    """
    control_sample_data = load_test_file(control_file)

    control_sample_scores = np.atleast_1d(
        evaluate_model(model_file, control_sample_data)
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


def score_directory(model_file: Path, graphs_dir: Path, out_csv: Path) -> None:
    """
    Run model scoring for all graph files in graphs_dir and write a CSV.

    Args:
        model_file: Path to the trained model checkpoint.
        graphs_dir: Directory containing processed graph files.
        out_csv: Path to output CSV that will store source, sequence, scores.
    """
    # clean previous output if present
    if out_csv.exists():
        out_csv.unlink()

    control_files = sorted(p for p in graphs_dir.iterdir() if p.is_file())

    # Do NOT use Pool when using CUDA
    for cf in control_files:
        sources, sequences, scores = process_control_file(cf, model_file)
        chunk_df = pd.DataFrame(
            {"source": sources, "sequence": sequences, "scores": scores},
            copy=False,
        )

        # append to CSV
        mode = "w" if not out_csv.exists() else "a"
        header = not out_csv.exists()
        chunk_df.to_csv(out_csv, mode=mode, index=False, header=header)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score TCR graphs with a trained model and write per sequence scores."
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        required=True,
        help="Path to the trained model file (best_model.pt).",
    )
    parser.add_argument(
        "--graphs-dir",
        type=Path,
        required=True,
        help="Directory containing processed graph files to score.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        required=True,
        help="Path to output CSV containing source, sequence, and scores.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    score_directory(
        model_file=args.model_file,
        graphs_dir=args.graphs_dir,
        out_csv=args.out_csv,
    )
