#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_improQ_model \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_gauss_res80_8to4in2_PD_cvol/model.pt --recon-model-name kengal_gauss \
--resolution 80 --maskconv-depth 3 --num-chans 16 --batch-size 16 --impro-model-name maskconv --num-epochs 50 \
--accelerations 8 --acquisition-steps 10 --report-interval 10 --num-target-rows 10 --lr 1e-4 --sample-rate 0.04 \
--seed 42 --eps-decay-rate 1 --start-eps 1.0 --num-workers 8 --in-chans 1 --lr-gamma 0.5 --center-volume \
--wandb --lr-step-size 20 --acquisition CORPD_FBK

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_improQ_model \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_gauss_res80_8to4in2_PD_cvol/model.pt --recon-model-name kengal_gauss \
--resolution 80 --maskconv-depth 3 --num-chans 16 --batch-size 16 --impro-model-name maskconv --num-epochs 50 \
--accelerations 8 --acquisition-steps 10 --report-interval 10 --num-target-rows 10 --lr 1e-4 --sample-rate 0.04 \
--seed 42 --eps-decay-rate 5 --start-eps 1.0 --num-workers 8 --in-chans 1 --lr-gamma 0.5 --center-volume \
--wandb --lr-step-size 20 --acquisition CORPD_FBK


CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_improQ_model \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_gauss_res80_8to4in2_PD_cvol/model.pt --recon-model-name kengal_gauss \
--resolution 80 --maskconv-depth 3 --num-chans 16 --batch-size 16 --impro-model-name maskconv --num-epochs 50 \
--accelerations 8 --acquisition-steps 10 --report-interval 10 --num-target-rows 10 --lr 5e-5 --sample-rate 0.04 \
--seed 42 --eps-decay-rate 1 --start-eps 0.5 --num-workers 8 --in-chans 1 --lr-gamma 0.5 --center-volume \
--wandb --lr-step-size 20 --acquisition CORPD_FBK

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_improQ_model \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_gauss_res80_8to4in2_PD_cvol/model.pt --recon-model-name kengal_gauss \
--resolution 80 --maskconv-depth 3 --num-chans 16 --batch-size 16 --impro-model-name maskconv --num-epochs 50 \
--accelerations 8 --acquisition-steps 10 --report-interval 10 --num-target-rows 10 --lr 5e-5 --sample-rate 0.04 \
--seed 42 --eps-decay-rate 1 --start-eps 1.0 --num-workers 8 --in-chans 1 --lr-gamma 0.5 --center-volume \
--wandb --lr-step-size 20 --acquisition CORPD_FBK

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_improQ_model \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_gauss_res80_8to4in2_PD_cvol/model.pt --recon-model-name kengal_gauss \
--resolution 80 --maskconv-depth 3 --num-chans 16 --batch-size 16 --impro-model-name maskconv --num-epochs 50 \
--accelerations 8 --acquisition-steps 10 --report-interval 10 --num-target-rows 10 --lr 5e-5 --sample-rate 0.04 \
--seed 42 --eps-decay-rate 5 --start-eps 1.0 --num-workers 8 --in-chans 1 --lr-gamma 0.5 --center-volume \
--wandb --lr-step-size 20 --acquisition CORPD_FBK


CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_improQ_model \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_gauss_res80_8to4in2_PD_cvol/model.pt --recon-model-name kengal_gauss \
--resolution 80 --maskconv-depth 3 --num-chans 16 --batch-size 16 --impro-model-name maskconv --num-epochs 50 \
--accelerations 8 --acquisition-steps 10 --report-interval 10 --num-target-rows 10 --lr 5e-5 --sample-rate 0.04 \
--seed 42 --eps-decay-rate 1 --start-eps 0.5 --num-workers 8 --in-chans 1 --lr-gamma 0.1 --center-volume \
--wandb --lr-step-size 40 --acquisition CORPD_FBK

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_improQ_model \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_gauss_res80_8to4in2_PD_cvol/model.pt --recon-model-name kengal_gauss \
--resolution 80 --maskconv-depth 3 --num-chans 16 --batch-size 16 --impro-model-name maskconv --num-epochs 50 \
--accelerations 8 --acquisition-steps 10 --report-interval 10 --num-target-rows 10 --lr 5e-5 --sample-rate 0.04 \
--seed 42 --eps-decay-rate 1 --start-eps 1.0 --num-workers 8 --in-chans 1 --lr-gamma 0.1 --center-volume \
--wandb --lr-step-size 40 --acquisition CORPD_FBK

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_improQ_model \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_gauss_res80_8to4in2_PD_cvol/model.pt --recon-model-name kengal_gauss \
--resolution 80 --maskconv-depth 3 --num-chans 16 --batch-size 16 --impro-model-name maskconv --num-epochs 50 \
--accelerations 8 --acquisition-steps 10 --report-interval 10 --num-target-rows 10 --lr 5e-5 --sample-rate 0.04 \
--seed 42 --eps-decay-rate 5 --start-eps 1.0 --num-workers 8 --in-chans 1 --lr-gamma 0.1 --center-volume \
--wandb --lr-step-size 40 --acquisition CORPD_FBK


CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_improQ_model \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_gauss_res80_8to4in2_PD_cvol/model.pt --recon-model-name kengal_gauss \
--resolution 80 --maskconv-depth 3 --num-chans 16 --batch-size 16 --impro-model-name maskconv --num-epochs 50 \
--accelerations 8 --acquisition-steps 10 --report-interval 10 --num-target-rows 10 --lr 1e-5 --sample-rate 0.04 \
--seed 42 --eps-decay-rate 1 --start-eps 0.5 --num-workers 8 --in-chans 1 --lr-gamma 0.5 --center-volume \
--wandb --lr-step-size 20 --acquisition CORPD_FBK

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_improQ_model \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_gauss_res80_8to4in2_PD_cvol/model.pt --recon-model-name kengal_gauss \
--resolution 80 --maskconv-depth 3 --num-chans 16 --batch-size 16 --impro-model-name maskconv --num-epochs 50 \
--accelerations 8 --acquisition-steps 10 --report-interval 10 --num-target-rows 10 --lr 1e-5 --sample-rate 0.04 \
--seed 42 --eps-decay-rate 1 --start-eps 1.0 --num-workers 8 --in-chans 1 --lr-gamma 0.5 --center-volume \
--wandb --lr-step-size 20 --acquisition CORPD_FBK

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_improQ_model \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_gauss_res80_8to4in2_PD_cvol/model.pt --recon-model-name kengal_gauss \
--resolution 80 --maskconv-depth 3 --num-chans 16 --batch-size 16 --impro-model-name maskconv --num-epochs 50 \
--accelerations 8 --acquisition-steps 10 --report-interval 10 --num-target-rows 10 --lr 1e-5 --sample-rate 0.04 \
--seed 42 --eps-decay-rate 5 --start-eps 1.0 --num-workers 8 --in-chans 1 --lr-gamma 0.5 --center-volume \
--wandb --lr-step-size 20 --acquisition CORPD_FBK