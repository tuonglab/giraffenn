#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=predict_gnn
#SBATCH --time=00:30:00
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --account=a_chrc
#SBATCH -e run.error
#SBATCH -o run.out

# Activate environment
source /scratch/project/tcr_ml/gnn_env/bin/activate

# Define paths and names
MODEL_PATH="/scratch/project/tcr_ml/gnn_release/MODELS/single_cell_soft_model.pt"
DATASET_NAME="theragen"
DATASET_PATH="/scratch/project/tcr_ml/gnn_release/test_data_v2/${DATASET_NAME}/processed"
OUT_CSV="theragen_soft_sc_sample_scores.csv"
# Run test script (with dataset name so scores folder is named correctly)
python /scratch/project/tcr_ml/gnn_release/research_scripts/test.py --graphs-dir "$DATASET_PATH" --model-file "$MODEL_PATH" --out-csv "$OUT_CSV"

