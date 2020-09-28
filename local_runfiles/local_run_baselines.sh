#!/usr/bin/env bash

source /home/timsey/anaconda3/bin/activate rim

# KNEE
#  BASE: oracle max batch size 8: 8.8GB, average_oracle 16: 9.9GB
#  LONG: oracle max batch size 8: 9.3GB, average_oracle 8: 9.3GB
CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp-dir /home/timsey/Projects/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--batch-size 64 --accelerations 8 --acquisition-steps 16 --sample-rate 0.5 --seed 0 --num-workers 8 --num-epochs 50 \
--acquisition None --center-volume True \
--wandb True --project mrimpro --original_setting True --low_res False --data-split test --model-type equispace_onesided

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp-dir /home/timsey/Projects/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--batch-size 64 --accelerations 8 --acquisition-steps 16 --sample-rate 0.5 --seed 0 --num-workers 8 --num-epochs 50 \
--acquisition None --center-volume True \
--wandb True --project mrimpro --original_setting True --low_res False --data-split test --model-type equispace_twosided

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp-dir /home/timsey/Projects/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--batch-size 8 --accelerations 8 --acquisition-steps 16 --sample-rate 0.5 --seed 0 --num-workers 8 --num-epochs 50 \
--acquisition None --center-volume True \
--wandb True --project mrimpro --original_setting True --low_res False --data-split test --model-type oracle

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp-dir /home/timsey/Projects/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--batch-size 16 --accelerations 8 --acquisition-steps 16 --sample-rate 0.5 --seed 0 --num-workers 8 --num-epochs 50 \
--acquisition None --center-volume True \
--wandb True --project mrimpro --original_setting True --low_res False --data-split test --model-type oracle_average

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp-dir /home/timsey/Projects/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--batch-size 64 --accelerations 8 --acquisition-steps 16 --sample-rate 0.5 --seed 0 --num-workers 8 --num-epochs 50 \
--acquisition None --center-volume True \
--wandb True --project mrimpro --original_setting True --low_res False --data-split test --model-type random

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp-dir /home/timsey/Projects/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--batch-size 64 --accelerations 8 --acquisition-steps 16 --sample-rate 0.5 --seed 0 --num-workers 8 --num-epochs 50 \
--acquisition None --center-volume True \
--wandb True --project mrimpro --original_setting True --low_res False --data-split test --model-type random

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp-dir /home/timsey/Projects/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--batch-size 64 --accelerations 8 --acquisition-steps 16 --sample-rate 0.5 --seed 0 --num-workers 8 --num-epochs 50 \
--acquisition None --center-volume True \
--wandb True --project mrimpro --original_setting True --low_res False --data-split test --model-type random

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp-dir /home/timsey/Projects/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--batch-size 64 --accelerations 8 --acquisition-steps 16 --sample-rate 0.5 --seed 0 --num-workers 8 --num-epochs 50 \
--acquisition None --center-volume True \
--wandb True --project mrimpro --original_setting True --low_res False --data-split test --model-type random

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp-dir /home/timsey/Projects/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--batch-size 64 --accelerations 8 --acquisition-steps 16 --sample-rate 0.5 --seed 0 --num-workers 8 --num-epochs 50 \
--acquisition None --center-volume True \
--wandb True --project mrimpro --original_setting True --low_res False --data-split test --model-type random



CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp-dir /home/timsey/Projects/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--batch-size 64 --accelerations 32 --acquisition-steps 28 --sample-rate 0.5 --seed 0 --num-workers 8 --num-epochs 50 \
--acquisition None --center-volume True \
--wandb True --project mrimpro --original_setting True --low_res False --data-split test --model-type equispace_onesided

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp-dir /home/timsey/Projects/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--batch-size 64 --accelerations 32 --acquisition-steps 28 --sample-rate 0.5 --seed 0 --num-workers 8 --num-epochs 50 \
--acquisition None --center-volume True \
--wandb True --project mrimpro --original_setting True --low_res False --data-split test --model-type equispace_twosided

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp-dir /home/timsey/Projects/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--batch-size 8 --accelerations 32 --acquisition-steps 28 --sample-rate 0.5 --seed 0 --num-workers 8 --num-epochs 50 \
--acquisition None --center-volume True \
--wandb True --project mrimpro --original_setting True --low_res False --data-split test --model-type oracle

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp-dir /home/timsey/Projects/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--batch-size 16 --accelerations 32 --acquisition-steps 28 --sample-rate 0.5 --seed 0 --num-workers 8 --num-epochs 50 \
--acquisition None --center-volume True \
--wandb True --project mrimpro --original_setting True --low_res False --data-split test --model-type oracle_average

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp-dir /home/timsey/Projects/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--batch-size 64 --accelerations 32 --acquisition-steps 28 --sample-rate 0.5 --seed 0 --num-workers 8 --num-epochs 50 \
--acquisition None --center-volume True \
--wandb True --project mrimpro --original_setting True --low_res False --data-split test --model-type random

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp-dir /home/timsey/Projects/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--batch-size 64 --accelerations 32 --acquisition-steps 28 --sample-rate 0.5 --seed 0 --num-workers 8 --num-epochs 50 \
--acquisition None --center-volume True \
--wandb True --project mrimpro --original_setting True --low_res False --data-split test --model-type random

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp-dir /home/timsey/Projects/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--batch-size 64 --accelerations 32 --acquisition-steps 28 --sample-rate 0.5 --seed 0 --num-workers 8 --num-epochs 50 \
--acquisition None --center-volume True \
--wandb True --project mrimpro --original_setting True --low_res False --data-split test --model-type random

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp-dir /home/timsey/Projects/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--batch-size 64 --accelerations 32 --acquisition-steps 28 --sample-rate 0.5 --seed 0 --num-workers 8 --num-epochs 50 \
--acquisition None --center-volume True \
--wandb True --project mrimpro --original_setting True --low_res False --data-split test --model-type random

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp-dir /home/timsey/Projects/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--batch-size 64 --accelerations 32 --acquisition-steps 28 --sample-rate 0.5 --seed 0 --num-workers 8 --num-epochs 50 \
--acquisition None --center-volume True \
--wandb True --project mrimpro --original_setting True --low_res False --data-split test --model-type random





# BRAIN
#  BASE: oracle max batch size 1: 6.5GB, average oracle 2: 9.9GB
#  LONG: oracle max batch size 1: 7.2GB, average_oracle 1: 7.2GB
CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/brain/ --exp-dir /home/timsey/Projects/mrimpro/brain_exp_results/ --resolution 256 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_brain_nonorig_highres256_8to4in2/model.pt --recon-model-name nounc \
--batch-size 64 --accelerations 8 --acquisition-steps 16 --sample-rate 0.2 --seed 0 --num-workers 8 --num-epochs 50  \
--acquisition None --center-volume False \
--wandb True --project mrimpro_brain --original_setting False --low_res False --data-split test --model-type equispace_onesided

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/brain/ --exp-dir /home/timsey/Projects/mrimpro/brain_exp_results/ --resolution 256 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_brain_nonorig_highres256_8to4in2/model.pt --recon-model-name nounc \
--batch-size 64 --accelerations 8 --acquisition-steps 16 --sample-rate 0.2 --seed 0 --num-workers 8 --num-epochs 50  \
--acquisition None --center-volume False \
--wandb True --project mrimpro_brain --original_setting False --low_res False --data-split test --model-type equispace_twosided

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/brain/ --exp-dir /home/timsey/Projects/mrimpro/brain_exp_results/ --resolution 256 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_brain_nonorig_highres256_8to4in2/model.pt --recon-model-name nounc \
--batch-size 1 --accelerations 8 --acquisition-steps 16 --sample-rate 0.2 --seed 0 --num-workers 8 --num-epochs 50  \
--acquisition None --center-volume False \
--wandb True --project mrimpro_brain --original_setting False --low_res False --data-split test --model-type oracle

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/brain/ --exp-dir /home/timsey/Projects/mrimpro/brain_exp_results/ --resolution 256 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_brain_nonorig_highres256_8to4in2/model.pt --recon-model-name nounc \
--batch-size 2 --accelerations 8 --acquisition-steps 16 --sample-rate 0.2 --seed 0 --num-workers 8 --num-epochs 50  \
--acquisition None --center-volume False \
--wandb True --project mrimpro_brain --original_setting False --low_res False --data-split test --model-type average_oracle

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/brain/ --exp-dir /home/timsey/Projects/mrimpro/brain_exp_results/ --resolution 256 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_brain_nonorig_highres256_8to4in2/model.pt --recon-model-name nounc \
--batch-size 64 --accelerations 8 --acquisition-steps 16 --sample-rate 0.2 --seed 0 --num-workers 8 --num-epochs 50  \
--acquisition None --center-volume False \
--wandb True --project mrimpro_brain --original_setting False --low_res False --data-split test --model-type random

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/brain/ --exp-dir /home/timsey/Projects/mrimpro/brain_exp_results/ --resolution 256 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_brain_nonorig_highres256_8to4in2/model.pt --recon-model-name nounc \
--batch-size 64 --accelerations 8 --acquisition-steps 16 --sample-rate 0.2 --seed 0 --num-workers 8 --num-epochs 50  \
--acquisition None --center-volume False \
--wandb True --project mrimpro_brain --original_setting False --low_res False --data-split test --model-type random

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/brain/ --exp-dir /home/timsey/Projects/mrimpro/brain_exp_results/ --resolution 256 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_brain_nonorig_highres256_8to4in2/model.pt --recon-model-name nounc \
--batch-size 64 --accelerations 8 --acquisition-steps 16 --sample-rate 0.2 --seed 0 --num-workers 8 --num-epochs 50  \
--acquisition None --center-volume False \
--wandb True --project mrimpro_brain --original_setting False --low_res False --data-split test --model-type random

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/brain/ --exp-dir /home/timsey/Projects/mrimpro/brain_exp_results/ --resolution 256 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_brain_nonorig_highres256_8to4in2/model.pt --recon-model-name nounc \
--batch-size 64 --accelerations 8 --acquisition-steps 16 --sample-rate 0.2 --seed 0 --num-workers 8 --num-epochs 50  \
--acquisition None --center-volume False \
--wandb True --project mrimpro_brain --original_setting False --low_res False --data-split test --model-type random

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/brain/ --exp-dir /home/timsey/Projects/mrimpro/brain_exp_results/ --resolution 256 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_brain_nonorig_highres256_8to4in2/model.pt --recon-model-name nounc \
--batch-size 64 --accelerations 8 --acquisition-steps 16 --sample-rate 0.2 --seed 0 --num-workers 8 --num-epochs 50  \
--acquisition None --center-volume False \
--wandb True --project mrimpro_brain --original_setting False --low_res False --data-split test --model-type random



CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/brain/ --exp-dir /home/timsey/Projects/mrimpro/brain_exp_results/ --resolution 256 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_brain_nonorig_highres256_8to4in2/model.pt --recon-model-name nounc \
--batch-size 64 --accelerations 32 --acquisition-steps 28 --sample-rate 0.2 --seed 0 --num-workers 8 --num-epochs 50  \
--acquisition None --center-volume False \
--wandb True --project mrimpro_brain --original_setting False --low_res False --data-split test --model-type equispace_onesided

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/brain/ --exp-dir /home/timsey/Projects/mrimpro/brain_exp_results/ --resolution 256 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_brain_nonorig_highres256_8to4in2/model.pt --recon-model-name nounc \
--batch-size 64 --accelerations 32 --acquisition-steps 28 --sample-rate 0.2 --seed 0 --num-workers 8 --num-epochs 50  \
--acquisition None --center-volume False \
--wandb True --project mrimpro_brain --original_setting False --low_res False --data-split test --model-type equispace_twosided

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/brain/ --exp-dir /home/timsey/Projects/mrimpro/brain_exp_results/ --resolution 256 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_brain_nonorig_highres256_8to4in2/model.pt --recon-model-name nounc \
--batch-size 1 --accelerations 32 --acquisition-steps 28 --sample-rate 0.2 --seed 0 --num-workers 8 --num-epochs 50  \
--acquisition None --center-volume False \
--wandb True --project mrimpro_brain --original_setting False --low_res False --data-split test --model-type oracle

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/brain/ --exp-dir /home/timsey/Projects/mrimpro/brain_exp_results/ --resolution 256 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_brain_nonorig_highres256_8to4in2/model.pt --recon-model-name nounc \
--batch-size 1 --accelerations 32 --acquisition-steps 28 --sample-rate 0.2 --seed 0 --num-workers 8 --num-epochs 50  \
--acquisition None --center-volume False \
--wandb True --project mrimpro_brain --original_setting False --low_res False --data-split test --model-type average_oracle

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/brain/ --exp-dir /home/timsey/Projects/mrimpro/brain_exp_results/ --resolution 256 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_brain_nonorig_highres256_8to4in2/model.pt --recon-model-name nounc \
--batch-size 64 --accelerations 32 --acquisition-steps 28 --sample-rate 0.2 --seed 0 --num-workers 8 --num-epochs 50  \
--acquisition None --center-volume False \
--wandb True --project mrimpro_brain --original_setting False --low_res False --data-split test --model-type random

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/brain/ --exp-dir /home/timsey/Projects/mrimpro/brain_exp_results/ --resolution 256 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_brain_nonorig_highres256_8to4in2/model.pt --recon-model-name nounc \
--batch-size 64 --accelerations 32 --acquisition-steps 28 --sample-rate 0.2 --seed 0 --num-workers 8 --num-epochs 50  \
--acquisition None --center-volume False \
--wandb True --project mrimpro_brain --original_setting False --low_res False --data-split test --model-type random

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/brain/ --exp-dir /home/timsey/Projects/mrimpro/brain_exp_results/ --resolution 256 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_brain_nonorig_highres256_8to4in2/model.pt --recon-model-name nounc \
--batch-size 64 --accelerations 32 --acquisition-steps 28 --sample-rate 0.2 --seed 0 --num-workers 8 --num-epochs 50  \
--acquisition None --center-volume False \
--wandb True --project mrimpro_brain --original_setting False --low_res False --data-split test --model-type random

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/brain/ --exp-dir /home/timsey/Projects/mrimpro/brain_exp_results/ --resolution 256 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_brain_nonorig_highres256_8to4in2/model.pt --recon-model-name nounc \
--batch-size 64 --accelerations 32 --acquisition-steps 28 --sample-rate 0.2 --seed 0 --num-workers 8 --num-epochs 50  \
--acquisition None --center-volume False \
--wandb True --project mrimpro_brain --original_setting False --low_res False --data-split test --model-type random

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/brain/ --exp-dir /home/timsey/Projects/mrimpro/brain_exp_results/ --resolution 256 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_brain_nonorig_highres256_8to4in2/model.pt --recon-model-name nounc \
--batch-size 64 --accelerations 32 --acquisition-steps 28 --sample-rate 0.2 --seed 0 --num-workers 8 --num-epochs 50  \
--acquisition None --center-volume False \
--wandb True --project mrimpro_brain --original_setting False --low_res False --data-split test --model-type random