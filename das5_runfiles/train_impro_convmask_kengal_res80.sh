#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.train_impro_model --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/lapace_res80_ch16_al_8to4in4/model.pt --recon-model-name kengal_laplace --resolution 80 \
--num-pools 4 --of-which-four-pools 0 --num-chans 16 --krow-batch-size 128 --batch-size 64 --impro-model-name convmask