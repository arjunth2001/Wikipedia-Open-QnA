#!/bin/bash
#SBATCH -A research
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH -o /scratch/arjun_wiki_db.txt
#SBATCH --job-name=wiki_db
source /home2/arjunth2001/miniconda3/etc/profile.d/conda.sh
conda activate legal
mkdir -p /scratch/arjunth2001/
scp -r arjunth2001@ada:'/share1/arjunth2001/data' /scratch/arjunth2001/
scp -r arjunth2001@ada:'/share1/arjunth2001/enwiki.xml.bz2' /scratch/arjunth2001/data
wikiextractor /scratch/arjunth2001/data/enwiki.xml.bz2 -o /scratch/arjunth2001/wiki_files --json
python write_to_db.py  /scratch/arjunth2001/wiki_files/  /scratch/arjunth2001/wiki.db
cd /scratch/arjunth2001/
scp -r wiki.db arjunth2001@ada:'/share1/arjunth2001/'