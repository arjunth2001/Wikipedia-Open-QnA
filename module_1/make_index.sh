#!/bin/bash
#SBATCH -A research
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH -o /scratch/arjun_wiki_full.txt
#SBATCH --job-name=wiki_full
source /home2/arjunth2001/miniconda3/etc/profile.d/conda.sh
conda activate legal
mkdir -p /scratch/arjunth2001/data
scp -r arjunth2001@ada:'/share1/arjunth2001/enwiki.xml.bz2' /scratch/arjunth2001/data
python make_index.py
cd /scratch/arjunth2001/data
rm -rf enwiki.xml.bz2
cd ..
scp -r data arjunth2001@ada:'/share1/arjunth2001/'