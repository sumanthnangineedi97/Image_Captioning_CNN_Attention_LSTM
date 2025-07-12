#!/bin/bash
#SBATCH --job-name=Image_captioning # Job name
#SBATCH --nodes=1 # Number of nodes
#SBATCH --ntasks-per-node=1 # Number of tasks per node
#SBATCH --cpus-per-task=16 # Number of CPUs per task
#SBATCH --mem=16G # Memory per node
#SBATCH --time=20:00:00 # Time limit hrs:min:sec
#SBATCH --output=AkhilaProjects/image_captioning/model_logs/output_Image_cap_attent_training_%j_%x_%A.log # Standard output and error log
#SBATCH --gpus-per-node=p100:2 # GPUs per node

module load anaconda3/2023.09-0
source activate dev
cd AkhilaProjects/image_captioning/

python3 src/train.py
