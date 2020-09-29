#!/bin/sh

#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --priority=TOP
#SBATCH --mem=10G
#SBATCH --verbose
#SBATCH --time 7-0:00:00
#SBATCH --job-name=brain

#SBATCH -D /home/tbbakke/mrimpro

echo "Running..."

source /home/tbbakke/anaconda3/bin/activate fastmri

nvidia-smi

# QR Brain 5 pool fc256 half-channel (hc).
CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.train_improQR_model_sweep \
--dataset fastmri --data-path /home/tbbakke/data/fastMRI/brain/ --exp-dir /home/tbbakke/mrimpro/brain_exp_results/ --resolution 256 \
--recon-model-checkpoint /home/tbbakke/fastMRI-shi/models/unet/al_brain_nonorig_highres256_8to4in2/model.pt \
--resume True --wandb True --project mrimpro_brain --run_id q4u8kpm8 \
--impro-model-checkpoint /home/tbbakke/mrimpro/brain_exp_results/res256_al28_accel[32]_convpool_nounc_k8_2020-09-27_04:11:42_VBVHJ/model.pt
