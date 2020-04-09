#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_improQS_model \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
--num-pools 4 --of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 512 --accelerations 8 --acquisition-steps 16 \
--report-interval 10 --num-target-rows 112 --lr 1e-4 --sample-rate 1 --seed 42 --eps-decay-rate 5 --num-workers 8 --in-chans 1 --center-volume \
--lr-gamma 0.1 --num-epochs 50 --scheduler-type step --lr-step-size 40 --start-eps 0 --start-tau 0.1 --tau-decay-rate 1 --do-dev-loss --pool-stride 1 \
--do-train-ssim --verbose 1 --wandb

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_improQS_model \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
--num-pools 4 --of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 512 --accelerations 128 --acquisition-steps 31 \
--report-interval 10 --num-target-rows 112 --lr 1e-4 --sample-rate 1 --seed 42 --eps-decay-rate 5 --num-workers 8 --in-chans 1 --center-volume \
--lr-gamma 0.1 --num-epochs 50 --scheduler-type step --lr-step-size 40 --start-eps 0 --start-tau 0.1 --tau-decay-rate 1 --do-dev-loss --pool-stride 1 \
--do-train-ssim --verbose 1 --wandb