#!/bin/bash

#SBATCH --job-name=Pt_W
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mem=2560M
#SBATCH --output=Pt_W.out
#SBATCH --account=132741350738

ternary_search -ll=0.7 -ul=1.3 -eps=0.02 -c="MLIPliquid -mlip=Grace -model=GRACE-2L-OMAT -dt=50"
