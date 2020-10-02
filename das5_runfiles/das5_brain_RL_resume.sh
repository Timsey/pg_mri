#!/bin/sh

#SBATCH --job-name=brain
#SBATCH --gres=gpu:4  # Hoeveel gpu heb je nodig?
#SBATCH -C GTX980Ti  # Welke gpus heb je nodig?

echo "Starting"

source /var/scratch/tbbakker/anaconda3/bin/activate fastmri
nvidia-smi

# 52 resume
CUDA_VISIBLE_DEVICES=0,1,2,3 HDF5_USE_FILE_LOCKING=FALSE PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/brain/ --exp-dir /var/scratch/tbbakker/mrimpro/brain256_results/ --resolution 256 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_brain_nonorig_highres256_8to4in2/model.pt \
--resume True --wandb True --project mrimpro_brain --run_id aq526fwo \
--impro-model-checkpoint /var/scratch/tbbakker/mrimpro/brain256_results/res256_al28_accel[32]_convpool_nounc_k8_2020-09-25_10:56:47_ITDYY/model.pt