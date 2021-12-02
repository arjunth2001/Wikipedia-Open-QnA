#!/bin/bash
#SBATCH -A research
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --exclude=gnode[01-45]
#SBATCH --mail-type=END,FAIL
#SBATCH -o /scratch/arjun_drqa.txt
#SBATCH --job-name=drqa
module load cuda/10.1
module load cudnn/7.6-cuda-10.0
module load TensorRT/7.2.2.3
source /home2/arjunth2001/miniconda3/etc/profile.d/conda.sh
conda activate py37
mkdir -p /scratch/arjunth2001/
scp -r arjunth2001@ada.iiit.ac.in:'/share1/arjunth2001/drqa_2/data' /scratch/arjunth2001
python train.py
rclone copy -P /scratch/arjun_drqa.txt  onedrive:/Adv_nlp_module_2