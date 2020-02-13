#!/usr/bin/env bash

# Maskconv baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_improQ_model \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_gauss_res80_8to4in2_PD_cvol/model.pt --recon-model-name kengal_gauss \
--resolution 80 --num-chans 16 --batch-size 16 --impro-model-name maskconv \
--accelerations 8 --acquisition-steps 10 --report-interval 10 --num-target-rows 10 --lr 1e-4 --sample-rate 0.04 \
--seed 42 --eps-decay-rate 1 --num-workers 8 --lr-step-size 20 --lr-gamma 10 --center-volume --wandb

# Center baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_improQ_model \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_gauss_res80_8to4in2_PD_cvol/model.pt --recon-model-name kengal_gauss \
--resolution 80 --batch-size 16 --accelerations 8 --acquisition-steps 10 --report-interval 10 --impro-model-name center \
--sample-rate 0.04 --seed 42 --num-workers 8  --center-volume --wandb

# Random baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_improQ_model \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_gauss_res80_8to4in2_PD_cvol/model.pt --recon-model-name kengal_gauss \
--resolution 80 --batch-size 16 --accelerations 8 --acquisition-steps 10 --report-interval 10 --impro-model-name random \
--sample-rate 0.04 --seed 42 --num-workers 8  --center-volume --wandb

# Convpoolmaskconv baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_improQ_model \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_gauss_res80_8to4in2_PD_cvol/model.pt --recon-model-name kengal_gauss \
--resolution 80 --num-pools 4 --of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpoolmaskconv --maskconv-depth 3 \
--fc-size 256 --accelerations 8 --acquisition-steps 10 --report-interval 10 --num-target-rows 10 --lr 2e-5 --sample-rate 0.04 \
--seed 42 --eps-decay-rate 1 --num-workers 8 --use-sensitivity --in-chans 3 --num-sens-samples 10 --lr-step-size 40 --lr-gamma 10 --center-volume --wandb


python -m src.train_improQ_model \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp-dir /home/timsey/Projects/mrimpro/results/ \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_gauss_res80_8to4in2_PD_cvol/model.pt --recon-model-name kengal_gauss \
--resolution 80 --num-pools 4 --of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpoolmaskconv --maskconv-depth 2 \
--fc-size 256 --accelerations 8 --acquisition-steps 10 --report-interval 10 --num-target-rows 10 --lr 2e-5 --sample-rate 0.04 \
--seed 42 --eps-decay-rate 1 --num-workers 8 --use-sensitivity --in-chans 3 --num-sens-samples 10 --lr-step-size 40 --lr-gamma 10 --center-volume --wandb