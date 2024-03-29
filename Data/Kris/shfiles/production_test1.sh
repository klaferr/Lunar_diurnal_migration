#!/bin/bash

#SBATCH -A bramsona
#SBATCH -t 0-48:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

python ../production_run.py -molecule "H2O" -noon 0 -time 24 -dt 0.25 -add 10 -num 1000 -dir "/home/klaferri/Desktop/Research/Lunar_diurnal_migration/Data/Kris/Results/Production/"

