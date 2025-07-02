#!/bin/bash
#SBATCH --time=10:00:00           # Request 10 hours of runtime
#SBATCH --account=st-tlwang-1-gpu     # Your CPU allocation code
#SBATCH --nodes=1                 # Request 1 node
#SBATCH --ntasks=1                # Request 1 task
#SBATCH --cpus-per-task=1         # Request 1 CPU per task
#SBATCH --mem=64G                 # Increased memory for GPU nodes
#SBATCH --gres=gpu:4              # Request 4 GPUs
#SBATCH --constraint=gpu_mem16GB  # Ensure each GPU has 16GB RAM
#SBATCH --job-name=aurora_job     # Job name for aurora_exp.py
#SBATCH -e slurm-%j.err           # Error file (%j will be replaced by the job ID)
#SBATCH -o slurm-%j.out           # Output file
#SBATCH --mail-user=jiangjing.gingercrystal@gmail.com  # Notification email
#SBATCH --mail-type=ALL           # Email notifications for all job events

# Load the necessary Python module (adjust the version if needed)
module load python/3.11.6

# Change to the directory where you submitted the job
cd $SLURM_SUBMIT_DIR

# Activate your previously created virtual environment
source /scratch/st-tlwang-1/jing/myenv/bin/activate

# Execute the Aurora experiment script
python script/aurora_exp.py "$@"