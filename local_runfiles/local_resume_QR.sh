#!/usr/bin/env bash

source /home/timsey/anaconda3/bin/activate rim

echo "---------------------------------"

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.train_improQR_model_sweep \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/singlecoil/ \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt \
--wandb True \
--resume True --run_id 2bgdrs2f --impro-model-checkpoint /home/timsey/Projects/mrimpro/exp_results/res128_al28_accel[32]_convpool_nounc_k8_2020-05-22_13:10:53/model.pt

#CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.train_improQR_model_sweep \
#--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/singlecoil/ \
#--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt \
#--wandb True \
#--resume True --run_id x8f8areg --impro-model-checkpoint /home/timsey/Projects/mrimpro/exp_results/res128_al28_accel[32]_convpool_nounc_k8_2020-05-22_13:10:25/model.pt
