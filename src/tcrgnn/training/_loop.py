from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from ._config import TrainConfig, TrainPaths


@dataclass
class TrainState:
    best_loss: float = float("inf")
    best_acc: float = 0.0
    patience_count: int = 0


def train_2(
    model: torch.nn.Module,
    samples: list[list[Data]],
    cfg: TrainConfig,
    save_path: TrainPaths,
    device: torch.device,
) -> None:
    """
    Sequence level training with bootstrapping loss.
    Bootstrapping uses a mix of hard labels and model predictions:

        t_boot = (1 - beta) * y + beta * p(x)

    This allows sequences in cancer samples that consistently look non-cancerous
    to have softer targets, so cancer logits are not all pushed high and
    control logits can stay low.
    """

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model.to(device)

    # Optional: still report class counts for debugging
    n_pos = 0
    n_neg = 0
    for sample in samples:
        for data in sample:
            y = data.y.view(-1)
            n_pos += int((y == 1).sum().item())
            n_neg += int((y == 0).sum().item())

    total = n_pos + n_neg
    print(f"sequences: total={total}, pos={n_pos}, neg={n_neg}")

    # Bootstrapping strength. You can move this into TrainConfig if you like.
    # beta = 0 uses pure hard labels
    # beta = 1 uses pure model predictions
    bootstrap_beta = getattr(cfg, "bootstrap_beta", 0.2)
    print(f"using bootstrap_beta={bootstrap_beta}")

    # Standard BCEWithLogitsLoss, no pos_weight here
    criterion = torch.nn.BCEWithLogitsLoss()

    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    pin = device.type == "cuda"
    loader = DataLoader(
        samples, batch_size=cfg.batch_size, shuffle=True, pin_memory=pin
    )

    state = TrainState()

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for sample in loader:
            for data in sample:
                data = data.to(device, non_blocking=pin)

                # logits for the cancer class
                logits = model(data.x, data.edge_index, data.batch)[:, 1]  # [B]

                # hard labels in {0, 1}
                raw_targets = data.y.float().view_as(logits)

                # current model probabilities, no grad through bootstrap targets
                with torch.no_grad():
                    probs = torch.sigmoid(logits)

                # bootstrapped targets
                # t_boot = (1 - beta) * y + beta * p
                targets = (1.0 - bootstrap_beta) * raw_targets + bootstrap_beta * probs

                optim.zero_grad()
                loss = criterion(logits, targets)
                loss.backward()
                optim.step()

                with torch.no_grad():
                    bs = targets.size(0)
                    total_loss += loss.item() * bs

                    # for accuracy, still use hard labels and standard 0.5 threshold
                    probs_eval = torch.sigmoid(logits)
                    pred = torch.round(probs_eval)

                    total_correct += (pred == raw_targets).sum().item()
                    total_samples += bs

        avg_loss = total_loss / max(total_samples, 1)
        acc = total_correct / max(total_samples, 1)
        print(f"epoch={epoch + 1} loss={avg_loss:.4f} acc={acc:.4f}")

        improved = False
        if avg_loss < state.best_loss - cfg.min_delta_loss:
            state.best_loss = avg_loss
            improved = True
        if acc > state.best_acc + cfg.min_delta_acc:
            state.best_acc = acc
            improved = True

        if improved:
            state.patience_count = 0
            torch.save(model.state_dict(), save_path.best_path)
        else:
            state.patience_count += 1
            if state.patience_count >= cfg.patience:
                print(f"Early stopping after no improvement for {cfg.patience} epochs")
                break


def train(
    model: torch.nn.Module,
    samples: list[list[Data]],
    cfg: TrainConfig,
    save_path: TrainPaths,
    device: torch.device,
) -> None:
    """
    Sequence level training with soft positive labels to avoid forcing
    all cancer sample sequences to be strong 1.
    """

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()

    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    pin = device.type == "cuda"
    loader = DataLoader(
        samples, batch_size=cfg.batch_size, shuffle=True, pin_memory=pin
    )

    state = TrainState()

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for sample in loader:
            for data in sample:
                data = data.to(device, non_blocking=pin)

                logits = model(data.x, data.edge_index, data.batch)[:, 1]  # [B]

                raw_targets = data.y.float().view_as(logits)

                # soft labels for cancer and control
                pos_target_value = 0.7
                neg_target_value = 0.05

                targets = torch.where(
                    raw_targets == 1.0,
                    torch.full_like(raw_targets, pos_target_value),
                    torch.full_like(raw_targets, neg_target_value),
                )

                optim.zero_grad()
                loss = criterion(logits, targets)
                loss.backward()
                optim.step()

                with torch.no_grad():
                    bs = targets.size(0)
                    total_loss += loss.item() * bs

                    probs = torch.sigmoid(logits)

                    hard_labels = raw_targets
                    pred = torch.round(probs)

                    total_correct += (pred == hard_labels).sum().item()
                    total_samples += bs

        avg_loss = total_loss / max(total_samples, 1)
        acc = total_correct / max(total_samples, 1)
        print(f"epoch={epoch + 1} loss={avg_loss:.4f} acc={acc:.4f}")

        improved = False
        if avg_loss < state.best_loss - cfg.min_delta_loss:
            state.best_loss = avg_loss
            improved = True
        if acc > state.best_acc + cfg.min_delta_acc:
            state.best_acc = acc
            improved = True

        if improved:
            state.patience_count = 0
            torch.save(model.state_dict(), save_path.best_path)
        else:
            state.patience_count += 1
            if state.patience_count >= cfg.patience:
                print(f"Early stopping after no improvement for {cfg.patience} epochs")
                break
