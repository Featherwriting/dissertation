#!/bin/bash
#SBATCH --mem=16G
#SBATCH --job-name=codegrokker_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

module load Anaconda3/5.3.0
module load cuDNN/7.6.4.38-gcccuda-2019b
source activate pytorch

python grok.py