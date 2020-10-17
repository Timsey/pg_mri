#!/usr/bin/env bash

source /home/timsey/anaconda3/bin/activate rim

CUDA_VISIBLE_DEVICES=0 python -m src.compute_snr \
--data_path /home/timsey/HDD/data/fastMRI/singlecoil/ \
--recon_model_checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt \
--base_policy_model_dir /home/timsey/Projects/mrimpro/refactor/ \
--policy_model_dir_list knee_res128_al16_accel[8]_k8_2020-10-16_17:36:55_BHONC knee_res128_al16_accel[8]_k8_2020-10-16_17:47:19_SWZTG