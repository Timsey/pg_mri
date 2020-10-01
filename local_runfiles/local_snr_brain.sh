#!/usr/bin/env bash

source /home/timsey/anaconda3/bin/activate rim

echo "---------------------------------"

# SNR BRAIN
CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.calculate_snr --dataset fastmri \
--data-path /home/timsey/HDD/data/fastMRI/brain/ --recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_brain_nonorig_highres256_8to4in2/model.pt \
--batch-size 16 --sample-rate 0.2 --seed 0 --num-workers 4 --num-trajectories 16 --batches-step 1 \
--style stoch --data_runs 3 --force_computation False --epochs 0 9 19 29 39 49 --base_impro_model_dir /home/timsey/Projects/mrimpro/brain_exp_results/ \
--impro_model_dir_list res256_al16_accel[8]_convpool_nounc_k8_2020-09-21_15:21:21_TUVLU res256_al16_accel[8]_convpool_nounc_k8_2020-09-21_15:03:43_HWQXH