#!/bin/bash

#SBATCH --job-name=Pt_W
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mem=2560M
#SBATCH --output=Pt_W.out
#SBATCH --account=132741350738

sqscal -e Pt,W -l BCC_A2,FCC_A1,LIQUID -lv 2 -sro -vib -mlip Grace -model GRACE-2L-OMAT
