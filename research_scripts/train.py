import os

from tcrgnn import TrainConfig, TrainPaths, train_model

cancer_dirs = [
    "/scratch/project/tcr_ml/gnn_release/dataset_v2/blood_tissue/processed",
    "/scratch/project/tcr_ml/gnn_release/dataset_v2/ccdi/processed",
    "/scratch/project/tcr_ml/gnn_release/dataset_v2/scTRB/processed",
    "/scratch/project/tcr_ml/gnn_release/dataset_v2/tumor_tissue/processed",
]
control_dirs = [
    "/scratch/project/tcr_ml/gnn_release/dataset_v2/curated/processed",
]

cfg = TrainConfig(
    epochs=500,
    batch_size=256,  # demonstration purpose only; use higher epochs for real training
)  # customise your training configuration here if neccessary

os.makedirs("custom_loss_model_val", exist_ok=True)
save_path = TrainPaths(
    model_dir="custom_loss_model_val",
    best_name="best_model.pt",
)

train_model(cancer_dirs, control_dirs, cfg, save_path)
