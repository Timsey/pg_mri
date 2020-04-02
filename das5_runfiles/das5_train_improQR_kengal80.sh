#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_improQR_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 80 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_gauss_res80_8to4in2_PD_cvol/model.pt --recon-model-name kengal_gauss \
--num-pools 4 --of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 \
--acquisition-steps 10 --report-interval 100 --num-target-rows 10 --lr 1e-4 --sample-rate 1 --seed 42 --num-workers 0 \
--in-chans 1 --lr-gamma 0.1 --num-epochs 50 --lr-step-size 40

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_improQR_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 80 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_gauss_res80_8to4in2_PD_cvol/model.pt --recon-model-name kengal_gauss \
--num-pools 8 --of-which-four-pools 0 --num-chans 12 --batch-size 16 --impro-model-name convpool --fc-size 2048 --accelerations 8 \
--acquisition-steps 10 --report-interval 100 --num-target-rows 10 --lr 1e-4 --sample-rate 0.04 --seed 42 --num-workers 8 \
--in-chans 1 --lr-gamma 0.1 --num-epochs 100 --lr-step-size 80 --pool-stride 2


CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_improQR_model_sweep \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp-dir /home/timsey/Projects/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 512 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--num-target-rows 20 --lr 1e-4 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --lr-gamma 0.1 --num-epochs 50 --lr-step-size 40 --num-pools 4 --pool-stride 1