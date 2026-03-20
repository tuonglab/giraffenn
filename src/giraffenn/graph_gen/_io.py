from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data


def list_edge_txts(root: Path | str) -> list[Path]:
    """
    Return all .txt edge list files inside the given directory (non-recursive).

    Parameters
    ----------
    root : str or pathlib.Path
        Directory to scan.

    Returns
    -------
    list[pathlib.Path]
        Paths of all files ending with .txt.
    """
    root = Path(root)
    return [p for p in root.iterdir() if p.suffix == ".txt" and p.is_file()]


def parse_edges(edge_file: Path | str) -> list[list[str]]:
    edge_file = Path(edge_file)
    with edge_file.open(encoding="utf-8", newline="") as f:
        return [line.strip().split() for line in f if line.strip()]


def _data_to_payload(g: Data) -> dict[str, Any]:
    """
    Convert a PyG Data object into a plain-Python/Tensor payload that is stable
    across PyTorch/PyG serialization changes.
    """
    payload: dict[str, Any] = {}

    # Data stores attributes in its internal mapping; keys() is stable.
    for k in g.keys():
        v = getattr(g, k)
        # torch.save supports tensors and basic python types well.
        payload[k] = v

    # Optional: keep some structural hints that can help debugging/future-proofing
    payload["_format"] = "pyg-data-dict-v1"
    return payload


def _payload_to_data(d: dict[str, Any]) -> Data:
    # Drop reserved keys
    d = {k: v for k, v in d.items() if not k.startswith("_")}
    return Data(**d)


def save_graphs_to_disk(graphs: Iterable[Data], out_file: str | Path) -> None:
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    payload: list[dict[str, Any]] = [_data_to_payload(g) for g in graphs]
    # Newer torch defaults to weights_only=True on load, but torch.save is the same.
    torch.save(payload, out_file)


def load_graphs_from_disk(
    in_file: str | Path, map_location: str | None = None
) -> list[Data]:
    in_file = Path(in_file)

    payload = torch.load(
        in_file, map_location=map_location
    )  # weights_only default is fine
    if not isinstance(payload, list):
        raise TypeError(f"Expected list payload, got {type(payload)}")

    return [_payload_to_data(d) for d in payload]


def clean_edge_files(edge_dir: Path | str) -> None:
    for p in list_edge_txts(edge_dir):
        if not p.exists():
            continue
        if p.stat().st_size == 0:
            print(f"[CLEANUP] Removing empty file: {p}")
            p.unlink()
