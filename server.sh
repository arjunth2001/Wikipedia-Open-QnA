#!/bin/bash
#SBATCH -A research
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --exclude=gnode[01-45]
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH -o /scratch/arjun_server.txt
#SBATCH --job-name=server
source /home2/arjunth2001/miniconda3/etc/profile.d/conda.sh
conda activate legal
mkdir -p /scratch/arjunth2001/data
mkdir -p /scratch/arjunth2001/dataset1
scp -r arjunth2001@ada:'/share1/arjunth2001/data' /scratch/arjunth2001/
scp -r arjunth2001@ada:'/share1/arjunth2001/enwiki.xml.bz2' /scratch/arjunth2001/data
scp -r arjunth2001@ada.iiit.ac.in:/share1/arjunth2001/wiki.db /scratch/arjunth2001
cd ~
jupyter-lab --no-browser --port=2001
