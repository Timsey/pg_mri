#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_improQS_model \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_gauss_res80_8to4in2_PD_cvol/model.pt --recon-model-name kengal_gauss \
--resolution 80 --num-pools 4 --of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convbottle --num-epochs 100 \
--fc-size 256 --out-chans 64 --accelerations 8 --acquisition-steps 10 --report-interval 10 --num-target-rows 10 --lr 1e-3 --sample-rate 0.04 \
--seed 42 --eps-decay-rate 1 --num-workers 8 --in-chans 1 --lr-gamma 10 --center-volume --start-eps 0.5 --tau 0.1 \
--wandb --scheduler-type multistep --lr-multi-step-size 40 --acquisition CORPD_FBK

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_improQS_model \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_gauss_res80_8to4in2_PD_cvol/model.pt --recon-model-name kengal_gauss \
--resolution 80 --num-pools 4 --of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convbottle --num-epochs 100 \
--fc-size 256 --out-chans 64 --accelerations 8 --acquisition-steps 10 --report-interval 10 --num-target-rows 10 --lr 1e-3 --sample-rate 0.04 \
--seed 42 --eps-decay-rate 5 --num-workers 8 --in-chans 1 --lr-gamma 10 --center-volume --start-eps 0.5 --tau 0.1 \
--wandb --scheduler-type multistep --lr-multi-step-size 40 --acquisition CORPD_FBK

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_improQS_model \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_gauss_res80_8to4in2_PD_cvol/model.pt --recon-model-name kengal_gauss \
--resolution 80 --num-pools 4 --of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convbottle --num-epochs 100 \
--fc-size 256 --out-chans 64 --accelerations 8 --acquisition-steps 10 --report-interval 10 --num-target-rows 10 --lr 1e-3 --sample-rate 0.04 \
--seed 42 --eps-decay-rate 10 --num-workers 8 --in-chans 1 --lr-gamma 10 --center-volume --start-eps 0.5 --tau 0.1 \
--wandb --scheduler-type multistep --lr-multi-step-size 40 --acquisition CORPD_FBK

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_improQS_model \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_gauss_res80_8to4in2_PD_cvol/model.pt --recon-model-name kengal_gauss \
--resolution 80 --num-pools 4 --of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --num-epochs 100 \
--fc-size 256 --out-chans 64 --accelerations 8 --acquisition-steps 10 --report-interval 10 --num-target-rows 10 --lr 1e-3 --sample-rate 0.04 \
--seed 42 --eps-decay-rate 1 --num-workers 8 --in-chans 1 --lr-gamma 10 --center-volume --start-eps 0.5 --tau 0.1 \
--wandb --scheduler-type multistep --lr-multi-step-size 40 --acquisition CORPD_FBK

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_improQS_model \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_gauss_res80_8to4in2_PD_cvol/model.pt --recon-model-name kengal_gauss \
--resolution 80 --num-pools 4 --of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --num-epochs 100 \
--fc-size 256 --out-chans 64 --accelerations 8 --acquisition-steps 10 --report-interval 10 --num-target-rows 10 --lr 1e-3 --sample-rate 0.04 \
--seed 42 --eps-decay-rate 5 --num-workers 8 --in-chans 1 --lr-gamma 10 --center-volume --start-eps 0.5 --tau 0.1 \
--wandb --scheduler-type multistep --lr-multi-step-size 40 --acquisition CORPD_FBK

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_improQS_model \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_gauss_res80_8to4in2_PD_cvol/model.pt --recon-model-name kengal_gauss \
--resolution 80 --num-pools 4 --of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --num-epochs 100 \
--fc-size 256 --out-chans 64 --accelerations 8 --acquisition-steps 10 --report-interval 10 --num-target-rows 10 --lr 1e-3 --sample-rate 0.04 \
--seed 42 --eps-decay-rate 10 --num-workers 8 --in-chans 1 --lr-gamma 10 --center-volume --start-eps 0.5 --tau 0.1 \
--wandb --scheduler-type multistep --lr-multi-step-size 40 --acquisition CORPD_FBK


CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_improQS_model \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_gauss_res80_8to4in2_PD_cvol/model.pt --recon-model-name kengal_gauss \
--resolution 80 --num-pools 4 --of-which-four-pools 0 --num-chans 32 --batch-size 16 --impro-model-name convpool --num-epochs 50 \
--fc-size 1024 --accelerations 8 --acquisition-steps 10 --report-interval 10 --num-target-rows 70 --lr 1e-4 --sample-rate 0.04 \
--seed 42 --num-workers 8 --in-chans 1 --lr-gamma 0.1 --center-volume --start-eps 0.0 --tau 0.1 --tau-decay-rate 2 \
--wandb --scheduler-type step --lr-step-size 40 --acquisition CORPD_FBK --do-dev-loss