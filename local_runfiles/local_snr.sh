#!/usr/bin/env bash

source /home/timsey/anaconda3/bin/activate rim

echo "---------------------------------"

# SNR for 5 of the knee gamma settings (best test model)
CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.calculate_snr --dataset fastmri \
--data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt \
--batch-size 16 --sample-rate 0.5 --seed 0 --num-workers 4 --num-trajectories 16 --acquisition None --center-volume True --batches-step 1 \
--original_setting True --low_res False --style stoch --data_runs 3 --force_computation True --epochs 0 9 19 29 39 49 --base_impro_model_dir /home/timsey/Projects/mrimpro/gamma_results_fix/ \
--impro_model_dir_list res128_al16_accel[8]_convpool_nounc_k8_2020-09-09_03:54:02_YWNLX \
res128_al16_accel[8]_convpool_nounc_k8_2020-09-07_04:04:11_KNBNY res128_al16_accel[8]_convpool_nounc_k8_2020-09-05_22:02:10_XAQXC \
res128_al16_accel[8]_convpool_nounc_k8_2020-09-05_21:16:23_SPDQS res128_al16_accel[8]_convpool_nounc_k8_2020-09-06_13:50:55_JAXUW \
res128_al16_accel[8]_convpool_nounc_k8_2020-09-07_04:22:39_WHLIG res128_al16_accel[8]_convpool_nounc_k8_2020-09-06_00:35:18_LTKUH \
res128_al16_accel[8]_convpool_nounc_k8_2020-09-02_16:39:26_PTMDD res128_al16_accel[8]_convpool_nounc_k8_2020-09-05_21:52:42_KOZOO \
res128_al16_accel[8]_convpool_nounc_k8_2020-09-04_15:02:47_EPPYS res128_al16_accel[8]_convpool_nounc_k8_2020-09-07_03:07:37_GEEHR \
res128_al28_accel[32]_convpool_nounc_k8_2020-09-02_18:59:21_SVPCK res128_al28_accel[32]_convpool_nounc_k8_2020-09-02_16:43:49_MNIJB


