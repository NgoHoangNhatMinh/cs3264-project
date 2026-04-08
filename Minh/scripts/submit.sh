#!/bin/bash
#!/bin/bash
#SBATCH --job-name=CS3264-Project-Training
#SBATCH --partition=gpu
#SBATCH --gpus=a100-40          # Requesting a 40GB A100
#SBATCH --time=01:00:00         # 1 hour limit (plenty for A100)
#SBATCH --mem=32G               # Request 32GB of System RAM
#SBATCH --output=training_%j.log # Log file for outputs/errors

module load anaconda3

conda activate cs3264-project-env

python model.py
