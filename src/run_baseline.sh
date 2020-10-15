#!/usr/bin/env bash

source /home/timsey/anaconda3/bin/activate rim

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.run_baseline_models \
--dataset brain --data_path /home/timsey/HDD/data/fastMRI/brain/ --exp_dir /home/timsey/Projects/mrimpro/refactor/ --resolution 256 \
--recon_model_checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_brain_nonorig_highres256_8to4in2/model.pt \
--batch_size 2 --accelerations 32 --acquisition_steps 28 --sample_rate 0.2 \
--center_volume False --wandb True --project mrimpro_brain --partition test --model_type oracle