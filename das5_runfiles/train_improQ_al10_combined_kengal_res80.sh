#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.train_improQ_model --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/lapace_res80_ch16_al_8to4in2_offcenter/model.pt --recon-model-name kengal_laplace --resolution 80 \
--num-pools 4 --num-chans 16 --batch-size 16 --impro-model-name convpool --of-which-four-pools 0 --accelerations 8 --acquisition-steps 10 \
--num-target-rows 80

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.train_improQ_model --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/lapace_res80_ch16_al_8to4in2_offcenter/model.pt --recon-model-name kengal_laplace --resolution 80 \
--num-pools 4 --num-chans 16 --batch-size 16 --impro-model-name convpool --of-which-four-pools 0 --accelerations 8 --acquisition-steps 10 \
--num-target-rows 10

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.train_improQ_model --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/lapace_res80_ch16_al_8to4in2_offcenter/model.pt --recon-model-name kengal_laplace --resolution 80 \
--num-pools 4 --num-chans 16 --batch-size 16 --impro-model-name convbottle --out-chans 64 --accelerations 8 --acquisition-steps 10 \
--num-target-rows 80

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.train_improQ_model --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/lapace_res80_ch16_al_8to4in2_offcenter/model.pt --recon-model-name kengal_laplace --resolution 80 \
--num-pools 4 --num-chans 16 --batch-size 16 --impro-model-name convbottle --out-chans 64 --accelerations 8 --acquisition-steps 10 \
--num-target-rows 10