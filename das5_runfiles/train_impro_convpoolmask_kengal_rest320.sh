#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.train_impro_model --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/lapace_res320_ch32_al_8to4in4/model.pt --recon-model-name kengal_laplace --resolution 320 \
--num-pools 4 --of-which-four-pools 2 --num-chans 32 --krow-batch-size 64 --batch-size 16 --impro-model-name convpoolmask