#!/usr/bin/env bash

source /home/timsey/anaconda3/bin/activate rim

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.train_policy \
--data_path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp_dir /home/timsey/Projects/mrimpro/refactor/ \
--recon_model_checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt \
--model_type greedy --project mri_refactor --wandb True --sample_rate 0.04

CUDA_VISIBLE_DEVICES=0 python -m src.train_policy \
--data_path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp_dir /home/timsey/Projects/mrimpro/refactor/ \
--recon_model_checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt \
--model_type nongreedy --project mri_refactor --wandb True \
--batch_size 4 --batches_step 4 --lr_gamma 0.5 --scheduler_type multistep --sample_rate 0.04