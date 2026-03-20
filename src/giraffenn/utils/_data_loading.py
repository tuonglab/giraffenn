from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data


def _is_pyg_payload(obj: Any) -> bool:
    """True if obj looks like a list of dict payloads representing PyG Data."""
    if not isinstance(obj, list):
        return False
    if len(obj) == 0:
        return True  # empty list is acceptable
    if not isinstance(obj[0], dict):
        return False
    # Heuristic keys typical of Data objects
    return "x" in obj[0] or "edge_index" in obj[0] or "y" in obj[0]


def _payload_to_data_list(payload: list[dict[str, Any]]) -> list[Data]:
    out: list[Data] = []
    for d in payload:
        # drop reserved keys if you add any later
        d2 = {k: v for k, v in d.items() if not k.startswith("_")}
        out.append(Data(**d2))
    return out


def load_graphs(
    file: str | Path,
    map_location: str | torch.device = "cpu",
    *,
    allow_unsafe_legacy: bool = True,
) -> list[Data] | Iterable[Data] | torch.Tensor:
    """
    Load a serialized graph bundle from disk.

    Robust behavior across PyTorch/PyG upgrades:
    - First attempts a "safe" load (weights_only=True semantics).
    - If that fails and allow_unsafe_legacy=True, falls back to unsafe pickle
      loading for legacy files that directly saved PyG Data objects.

    Args:
        file: Path to a serialized graph file.
        map_location: Passed through to torch.load.
        allow_unsafe_legacy: If True, permits loading legacy pickled objects
            by setting weights_only=False on fallback. Only enable for trusted files.

    Returns:
        Commonly:
            - list[Data]
            - iterable of Data
            - PyTorch tensor
    """
    file = Path(file)

    # 1) Safe path (PyTorch default weights_only=True in >=2.6)
    try:
        obj = torch.load(str(file), map_location=map_location)
    except Exception:
        # 2) Legacy fallback for trusted files
        if not allow_unsafe_legacy:
            raise
        obj = torch.load(str(file), map_location=map_location, weights_only=False)

    # If the file used the "safe payload" format, reconstruct Data objects:
    if _is_pyg_payload(obj):
        return _payload_to_data_list(obj)  # type: ignore[arg-type]

    return obj


def load_train_data(
    cancer_dirs: list[str | os.PathLike],
    control_dirs: list[str | os.PathLike],
) -> list[list[Data]]:
    """
    Load training samples from cancer and control directories.

    Each directory is expected to contain files where each file loads
    to a list of PyTorch Geometric Data objects. Each *file* becomes
    one sample in the training set.

    Args:
        cancer_dirs: Directories for positive samples.
        control_dirs: Directories for negative samples.

    Returns:
        A list of samples, where each sample is a list[Data].
    """
    training_set: list[list[Data]] = []

    for d in cancer_dirs:
        if os.path.isdir(d):
            for fname in os.listdir(d):
                graphs = load_graphs(os.path.join(d, fname))
                training_set.append(graphs)

    for d in control_dirs:
        if os.path.isdir(d):
            for fname in os.listdir(d):
                graphs = load_graphs(os.path.join(d, fname))
                training_set.append(graphs)

    return training_set


def load_test_file(file_path: str | Path) -> list[Data] | Iterable[Data]:
    """
    Load a test graph bundle from a file.

    Args:
        file_path: Path to a saved graph list.

    Returns:
        List (or iterable) of Data objects.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    file_path = Path(file_path)

    if file_path.is_file():
        return load_graphs(file_path)

    raise FileNotFoundError(f"Test file not found: {file_path}")
