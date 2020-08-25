#!/usr/bin/env bash

source /home/timsey/anaconda3/bin/activate rim

echo "---------------------------------"

# SNR for 5 of the knee gamma settings (best test model)
CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.calculate_snr --dataset fastmri \
--data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt \
--batch-size 16 --sample-rate 0.5 --seed 0 --num-workers 4 --num-trajectories 16 --acquisition None --center-volume True --batches-step 1 \
--original_setting True --low_res False --style stoch --data_runs 3 --force_computation False --epochs 0 9 19 29 39 49 \
--impro_model_dir_list res128_al16_accel[8]_convpool_nounc_k8_2020-08-22_01:00:23_XXRJZ res128_al16_accel[8]_convpool_nounc_k8_2020-08-23_05:57:01_IIQRM \
res128_al16_accel[8]_convpool_nounc_k8_2020-08-21_09:05:20_YFHWB res128_al16_accel[8]_convpool_nounc_k8_2020-08-22_12:56:28_FPWBU \
res128_al16_accel[8]_convpool_nounc_k8_2020-08-20_17:52:34_EUNVB res128_al16_accel[8]_convpool_nounc_k8_2020-08-23_06:55:48_JJHOR \
--base_impro_model_dir /home/timsey/Projects/mrimpro/gamma_results/