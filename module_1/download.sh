#!/bin/bash
#SBATCH -A research
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:0
#SBATCH --mem=8G
#SBATCH --exclude=gnode[01-45]
#SBATCH --mail-type=END,FAIL
#SBATCH -o /scratch/arjun_wiki.txt
#SBATCH --job-name=wiki
mkdir /scratch/arjunth2001
cd /scratch/arjunth2001
curl https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2 --output enwiki.xml.bz2
scp -r enwiki.xml.bz2 arjunth2001@ada:'/share1/arjunth2001/'