#!/bin/bash
#SBATCH -A research
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH -o /scratch/arjun_drqa_pre.txt
#SBATCH --job-name=drqa_pre
source /home2/arjunth2001/miniconda3/etc/profile.d/conda.sh
conda activate py37
mkdir -p /scratch/arjunth2001/drqa
scp -r arjunth2001@ada.iiit.ac.in:'/share1/arjunth2001/glove.840B.300d.zip' /scratch/arjunth2001/
unzip /scratch/arjunth2001/glove.840B.300d.zip -d /scratch/arjunth2001/
papermill  --request-save-on-cell-execute --log-output --log-level INFO --progress-bar preprocess.ipynb /scratch/arjunth2001/out.ipynb
scp -r /scratch/arjunth2001/drqa arjunth2001@ada.iiit.ac.in:/share1/arjunth2001/