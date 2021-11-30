#!/bin/bash
#SBATCH -A research
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH -o /scratch/arjun_wiki_1.txt
#SBATCH --job-name=wiki1
source /home2/arjunth2001/miniconda3/etc/profile.d/conda.sh
conda activate legal
mkdir -p /scratch/arjunth2001/data
mkdir -p /scratch/arjunth2001/dataset1
scp -r arjunth2001@ada:'/share1/arjunth2001/data' /scratch/arjunth2001/
scp -r arjunth2001@ada:'/share1/arjunth2001/enwiki.xml.bz2' /scratch/arjunth2001/data
scp -r arjunth2001@ada.iiit.ac.in:/share1/arjunth2001/wiki.db /scratch/arjunth2001
python run_squad1.py
scp -r /scratch/arjunth2001/dataset1 arjunth2001@ada.iiit.ac.in:/share1/arjunth2001/