#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_gnn
#SBATCH --time=04:30:00
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --account=a_kelvin_tuong
#SBATCH -e train2.error
#SBATCH -o train2.out

source /scratch/project/tcr_ml/gnn_env/bin/activate
python train.py