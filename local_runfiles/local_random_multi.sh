#!/usr/bin/env bash

source /home/timsey/anaconda3/bin/activate rim

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models --dataset fastmri \
--data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp-dir /home/timsey/Projects/mrimpro/results/ \
--resolution 128 --recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt \
--recon-model-name nounc --batch-size 64 --accelerations 8 --acquisition-steps 16 --sample-rate 0.5 --seed 0 --num-workers 8 \
--center-volume --num-epochs 1 --model-type random --acquisition None --data-range volume --use-data-state False \
--data-split test --wandb

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models --dataset fastmri \
--data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp-dir /home/timsey/Projects/mrimpro/results/ \
--resolution 128 --recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt \
--recon-model-name nounc --batch-size 64 --accelerations 8 --acquisition-steps 16 --sample-rate 0.5 --seed 0 --num-workers 8 \
--center-volume --num-epochs 1 --model-type random --acquisition None --data-range volume --use-data-state False \
--data-split test --wandb

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models --dataset fastmri \
--data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp-dir /home/timsey/Projects/mrimpro/results/ \
--resolution 128 --recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt \
--recon-model-name nounc --batch-size 64 --accelerations 8 --acquisition-steps 16 --sample-rate 0.5 --seed 0 --num-workers 8 \
--center-volume --num-epochs 1 --model-type random --acquisition None --data-range volume --use-data-state False \
--data-split test --wandb

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models --dataset fastmri \
--data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp-dir /home/timsey/Projects/mrimpro/results/ \
--resolution 128 --recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt \
--recon-model-name nounc --batch-size 64 --accelerations 8 --acquisition-steps 16 --sample-rate 0.5 --seed 0 --num-workers 8 \
--center-volume --num-epochs 1 --model-type random --acquisition None --data-range volume --use-data-state False \
--data-split test --wandb

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models --dataset fastmri \
--data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp-dir /home/timsey/Projects/mrimpro/results/ \
--resolution 128 --recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt \
--recon-model-name nounc --batch-size 64 --accelerations 8 --acquisition-steps 16 --sample-rate 0.5 --seed 0 --num-workers 8 \
--center-volume --num-epochs 1 --model-type random --acquisition None --data-range volume --use-data-state False \
--data-split test --wandb




CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models --dataset fastmri \
--data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp-dir /home/timsey/Projects/mrimpro/results/ \
--resolution 128 --recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt \
--recon-model-name nounc --batch-size 64 --accelerations 32 --acquisition-steps 28 --sample-rate 0.5 --seed 0 --num-workers 8 \
--center-volume --num-epochs 1 --model-type random --acquisition None --data-range volume --use-data-state False \
--data-split test --wandb

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models --dataset fastmri \
--data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp-dir /home/timsey/Projects/mrimpro/results/ \
--resolution 128 --recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt \
--recon-model-name nounc --batch-size 64 --accelerations 32 --acquisition-steps 28 --sample-rate 0.5 --seed 0 --num-workers 8 \
--center-volume --num-epochs 1 --model-type random --acquisition None --data-range volume --use-data-state False \
--data-split test --wandb

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models --dataset fastmri \
--data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp-dir /home/timsey/Projects/mrimpro/results/ \
--resolution 128 --recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt \
--recon-model-name nounc --batch-size 64 --accelerations 32 --acquisition-steps 28 --sample-rate 0.5 --seed 0 --num-workers 8 \
--center-volume --num-epochs 1 --model-type random --acquisition None --data-range volume --use-data-state False \
--data-split test --wandb

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models --dataset fastmri \
--data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp-dir /home/timsey/Projects/mrimpro/results/ \
--resolution 128 --recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt \
--recon-model-name nounc --batch-size 64 --accelerations 32 --acquisition-steps 28 --sample-rate 0.5 --seed 0 --num-workers 8 \
--center-volume --num-epochs 1 --model-type random --acquisition None --data-range volume --use-data-state False \
--data-split test --wandb

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models --dataset fastmri \
--data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp-dir /home/timsey/Projects/mrimpro/results/ \
--resolution 128 --recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt \
--recon-model-name nounc --batch-size 64 --accelerations 32 --acquisition-steps 28 --sample-rate 0.5 --seed 0 --num-workers 8 \
--center-volume --num-epochs 1 --model-type random --acquisition None --data-range volume --use-data-state False \
--data-split test --wandb