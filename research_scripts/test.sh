#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --job-name=predict_gnn
#SBATCH --time=3:30:00
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --account=a_kelvin_tuong
#SBATCH -e test.error
#SBATCH -o test.out

source /scratch/project/tcr_ml/gnn_env/bin/activate
MODEL="/scratch/project/tcr_ml/gnn_release/research_scripts/boltz_soft_label_model/best_model.pt"
GRAPHS="/scratch/project/tcr_ml/gnn_release/test_data_v2/val_control/processed"
OUTCSV="/scratch/project/tcr_ml/gnn_release/research_scripts/boltz_scores/val_control_sample_scores.csv"

python test.py --model-file "$MODEL" --graphs-dir "$GRAPHS" --out-csv "$OUTCSV"
