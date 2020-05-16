#!/usr/bin/env bash

source /home/amlab-root/anaconda3/bin/activate fastmri

echo "---------------------------------"

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /home/amlab-root/timu/mnt/HDD/data/fastMRI/singlecoil/ --exp-dir /home/amlab-root/timu/mnt/Projects/mrimpro/var_results/RL_self/ --resolution 128 \
--recon-model-checkpoint /home/amlab-root/timu/mnt/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 4 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 1e-4 --sample-rate 0.1 --seed 0 --num-workers 4 --in-chans 1 --num-epochs 30 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 8 --num-dev-trajectories 8 --greedy False --data-range volume --baseline-type self \
--scheduler-type multistep --lr-multi-step-size 10 20 --lr-gamma .1 --acquisition None --center-volume True --batches-step 4 \
--wandb True --do-train-ssim True

echo "---------------------------------"

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /home/amlab-root/timu/mnt/HDD/data/fastMRI/singlecoil/ --exp-dir /home/amlab-root/timu/mnt/Projects/mrimpro/var_results/RL_self/ --resolution 128 \
--recon-model-checkpoint /home/amlab-root/timu/mnt/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 4 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 1e-4 --sample-rate 0.1 --seed 0 --num-workers 4 --in-chans 1 --num-epochs 30 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 8 --num-dev-trajectories 8 --greedy False --data-range volume --baseline-type self \
--scheduler-type multistep --lr-multi-step-size 10 20 --lr-gamma .1 --acquisition None --center-volume True --batches-step 4 \
--wandb True --do-train-ssim True

echo "---------------------------------"

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /home/amlab-root/timu/mnt/HDD/data/fastMRI/singlecoil/ --exp-dir /home/amlab-root/timu/mnt/Projects/mrimpro/var_results/RL_self/ --resolution 128 \
--recon-model-checkpoint /home/amlab-root/timu/mnt/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 4 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 1e-4 --sample-rate 0.1 --seed 0 --num-workers 4 --in-chans 1 --num-epochs 30 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 8 --num-dev-trajectories 8 --greedy False --data-range volume --baseline-type self \
--scheduler-type multistep --lr-multi-step-size 10 20 --lr-gamma .1 --acquisition None --center-volume True --batches-step 4 \
--wandb True --do-train-ssim True

echo "---------------------------------"

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /home/amlab-root/timu/mnt/HDD/data/fastMRI/singlecoil/ --exp-dir /home/amlab-root/timu/mnt/Projects/mrimpro/var_results/RL_self/ --resolution 128 \
--recon-model-checkpoint /home/amlab-root/timu/mnt/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 4 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 1e-4 --sample-rate 0.1 --seed 0 --num-workers 4 --in-chans 1 --num-epochs 30 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 8 --num-dev-trajectories 8 --greedy False --data-range volume --baseline-type self \
--scheduler-type multistep --lr-multi-step-size 10 20 --lr-gamma .1 --acquisition None --center-volume True --batches-step 4 \
--wandb True --do-train-ssim True

echo "---------------------------------"

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /home/amlab-root/timu/mnt/HDD/data/fastMRI/singlecoil/ --exp-dir /home/amlab-root/timu/mnt/Projects/mrimpro/var_results/RL_self/ --resolution 128 \
--recon-model-checkpoint /home/amlab-root/timu/mnt/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 4 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 1e-4 --sample-rate 0.1 --seed 0 --num-workers 4 --in-chans 1 --num-epochs 30 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 8 --num-dev-trajectories 8 --greedy False --data-range volume --baseline-type self \
--scheduler-type multistep --lr-multi-step-size 10 20 --lr-gamma .1 --acquisition None --center-volume True --batches-step 4 \
--wandb True --do-train-ssim True

echo "---------------------------------"

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /home/amlab-root/timu/mnt/HDD/data/fastMRI/singlecoil/ --exp-dir /home/amlab-root/timu/mnt/Projects/mrimpro/var_results/RL_self/ --resolution 128 \
--recon-model-checkpoint /home/amlab-root/timu/mnt/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 4 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 1e-4 --sample-rate 0.1 --seed 0 --num-workers 4 --in-chans 1 --num-epochs 30 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 8 --num-dev-trajectories 8 --greedy False --data-range volume --baseline-type self \
--scheduler-type multistep --lr-multi-step-size 10 20 --lr-gamma .1 --acquisition None --center-volume True --batches-step 4 \
--wandb True --do-train-ssim True

echo "---------------------------------"

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /home/amlab-root/timu/mnt/HDD/data/fastMRI/singlecoil/ --exp-dir /home/amlab-root/timu/mnt/Projects/mrimpro/var_results/RL_self/ --resolution 128 \
--recon-model-checkpoint /home/amlab-root/timu/mnt/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 4 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 1e-4 --sample-rate 0.1 --seed 0 --num-workers 4 --in-chans 1 --num-epochs 30 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 8 --num-dev-trajectories 8 --greedy False --data-range volume --baseline-type self \
--scheduler-type multistep --lr-multi-step-size 10 20 --lr-gamma .1 --acquisition None --center-volume True --batches-step 4 \
--wandb True --do-train-ssim True

echo "---------------------------------"

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /home/amlab-root/timu/mnt/HDD/data/fastMRI/singlecoil/ --exp-dir /home/amlab-root/timu/mnt/Projects/mrimpro/var_results/RL_self/ --resolution 128 \
--recon-model-checkpoint /home/amlab-root/timu/mnt/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 4 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 1e-4 --sample-rate 0.1 --seed 0 --num-workers 4 --in-chans 1 --num-epochs 30 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 8 --num-dev-trajectories 8 --greedy False --data-range volume --baseline-type self \
--scheduler-type multistep --lr-multi-step-size 10 20 --lr-gamma .1 --acquisition None --center-volume True --batches-step 4 \
--wandb True --do-train-ssim True

echo "---------------------------------"

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /home/amlab-root/timu/mnt/HDD/data/fastMRI/singlecoil/ --exp-dir /home/amlab-root/timu/mnt/Projects/mrimpro/var_results/RL_self/ --resolution 128 \
--recon-model-checkpoint /home/amlab-root/timu/mnt/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 4 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 1e-4 --sample-rate 0.1 --seed 0 --num-workers 4 --in-chans 1 --num-epochs 30 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 8 --num-dev-trajectories 8 --greedy False --data-range volume --baseline-type self \
--scheduler-type multistep --lr-multi-step-size 10 20 --lr-gamma .1 --acquisition None --center-volume True --batches-step 4 \
--wandb True --do-train-ssim True

echo "---------------------------------"

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /home/amlab-root/timu/mnt/HDD/data/fastMRI/singlecoil/ --exp-dir /home/amlab-root/timu/mnt/Projects/mrimpro/var_results/RL_self/ --resolution 128 \
--recon-model-checkpoint /home/amlab-root/timu/mnt/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 4 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 1e-4 --sample-rate 0.1 --seed 0 --num-workers 4 --in-chans 1 --num-epochs 30 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 8 --num-dev-trajectories 8 --greedy False --data-range volume --baseline-type self \
--scheduler-type multistep --lr-multi-step-size 10 20 --lr-gamma .1 --acquisition None --center-volume True --batches-step 4 \
--wandb True --do-train-ssim True