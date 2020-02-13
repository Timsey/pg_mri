#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.train_improQ_model \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp-dir /home/timsey/Projects/mrimpro/results/ \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_gauss_res80_8to4in2_PD_cvol/model.pt --recon-model-name kengal_gauss \
--resolution 80 --num-pools 4 --of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpoolmaskconv \
--out-chans 64 --fc-size 256 --accelerations 8 --acquisition-steps 10 --report-interval 10 --num-target-rows 10 --lr 2e-5 --sample-rate 0.04 \
--seed 42 --eps-decay-rate 5 --num-workers 8 --use-sensitivity --in-chans 3 --num-sens-samples 10 --lr-step-size 10 --lr-gamma 5 --center-volume --wandb