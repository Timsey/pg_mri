#!/bin/sh

#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --priority=TOP
#SBATCH --mem=10G
#SBATCH --verbose
#SBATCH --time 5-0:00:00
#SBATCH --job-name=greedy

#SBATCH -D /home/tbbakke/mrimpro

echo "Running..."

source /home/tbbakke/anaconda3/bin/activate fastmri

nvidia-smi

# Do your stuff

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models \
--dataset fastmri --data-path /home/tbbakke/data/fastMRI/brain/ --exp-dir /home/tbbakke/mrimpro/brain_exp_results/ --resolution 256 \
--recon-model-checkpoint /home/tbbakke/fastMRI-shi/models/unet/al_brain_nonorig_highres256_8to4in2/model.pt --recon-model-name nounc \
--batch-size 1 --accelerations 32 --acquisition-steps 28 --sample-rate 0.2 --seed 0 --num-workers 8 --num-epochs 50  \
--acquisition None --center-volume False \
--wandb True --project mrimpro_brain --original_setting False --low_res False --data-split test --model-type oracle