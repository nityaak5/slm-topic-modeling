#!/bin/bash
#SBATCH --job-name=stance_models
#SBATCH --partition=GPU
#SBATCH --nodelist=imperial
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=n.kalra@tilburguniversity.edu

set -euo pipefail

PROJECT_DIR="$HOME/slm-topic-modeling/topic_modeling_paper"
cd "$PROJECT_DIR"
source venv/bin/activate

mkdir -p logs
python run_from_csv.py "${SLURM_ARRAY_TASK_ID}"
