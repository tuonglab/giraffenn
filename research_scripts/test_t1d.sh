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
#SBATCH -e t1d.error
#SBATCH -o t1d.out

source /scratch/project/tcr_ml/gnn_env/bin/activate
python test_t1d.py