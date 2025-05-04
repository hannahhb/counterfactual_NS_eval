#!/bin/bash
#
#SBATCH -A punim0478
#SBATCH --time=0-3:00:00
#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=70G
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --job-name=folio_eval

# Load required modules
module load CUDA
source /home/${USER}/.bashrc
source activate neurosymbolic

export PATH=/data/projects/punim0478/bansaab/LADR-2009-11A/bin:$PATH
export HF_HOME="/data/projects/punim0478/bansaab"
export HF_TOKEN="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"
export PYTHONPATH="$PYTHONPATH:$(pwd)"  # Add current directory to Python path

echo "Running on node: $(hostname)"
echo "GPU information: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start time: $(date)"

# Launch multiple process python code
time python main.py

echo "End time: $(date)"
