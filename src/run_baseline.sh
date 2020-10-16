#!/usr/bin/env bash

source /home/timsey/anaconda3/bin/activate rim

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models \
--dataset knee --data_path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp_dir /home/timsey/Projects/mrimpro/refactor/ --resolution 128 \
--recon_model_checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt \
--batch_size 8 --accelerations 8 --acquisition_steps 16 --sample_rate 0.5 \
--center_volume True --wandb True --project mrimpro_brain --partition test --model_type average_oracle