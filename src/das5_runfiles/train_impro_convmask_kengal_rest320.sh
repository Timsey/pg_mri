#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python ./src/train_impro_model.py --data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp-dir /home/timsey/Projects/mrimpro/results/ \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/lapace_res320_ch32_al_8to4in4/best_model.pt --recon-model-name kengal_laplace --resolution 320 \
--num-pools 6 --num-chans 32 --krow-batch-size 64 --batch-size 16 --impro-model-name convmask