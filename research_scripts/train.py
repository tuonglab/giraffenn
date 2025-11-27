import os

from tcrgnn import TrainConfig, TrainPaths, train_model

cancer_dirs = [
    # "/scratch/project/tcr_ml/gnn_release/dataset_v2/blood_tissue/processed",
    # "/scratch/project/tcr_ml/gnn_release/dataset_v2/ccdi/processed",
    "/scratch/project/tcr_ml/gnn_release/dataset_v2/scTRB/processed",
    # "/scratch/project/tcr_ml/gnn_release/dataset_v2/tumor_tissue/processed",
]
control_dirs = [
    # "/scratch/project/tcr_ml/gnn_release/dataset_v2/curated/processed",
    # "/scratch/project/tcr_ml/gnn_release/dataset_v2/control/processed"
    "/scratch/project/tcr_ml/gnn_release/dataset_v2/single_cell_control/processed"
]

cfg = TrainConfig(
    epochs=500,
    batch_size=256,  # demonstration purpose only; use higher epochs for real training
)  # customise your training configuration here if neccessary

os.makedirs("sc_soft_label_model", exist_ok=True)
save_path = TrainPaths(
    model_dir="sc_soft_label_model",
    best_name="best_model.pt",
)

train_model(cancer_dirs, control_dirs, cfg, save_path)
