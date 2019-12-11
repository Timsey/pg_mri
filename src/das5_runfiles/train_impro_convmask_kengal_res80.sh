#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python ./src/train_impro_model.py --data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp-dir /home/timsey/Projects/mrimpro/results/ \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/lapace_res80_ch16_al_8to4in4/best_model.pt --recon-model-name kengal_laplace --resolution 80 \
--num-pools 4 --num-chans 16 --krow-batch-size 128 --batch-size 64 --impro-model-name convmask