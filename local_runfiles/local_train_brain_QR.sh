#!/usr/bin/env bash

source /home/timsey/anaconda3/bin/activate rim

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.train_improQR_model_sweep \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/brain/ --exp-dir /home/timsey/Projects/mrimpro/brain_exp_results/ --resolution 256 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_brain_nonorig_highres256_8to4in2/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 8 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 1000 \
--num-target-rows 8 --lr 5e-5 --sample-rate 0.2 --seed 0 --num-workers 4 --in-chans 1 --lr-gamma 0.1 --num-epochs 50 --num-pools 5 --pool-stride 1 \
--estimator wr --acq_strat sample --acquisition None --center-volume False --lr-step-size 40 --wandb True --do-train-ssim False --num-test-trajectories 1 \
--project mrimpro_brain --original_setting False --low_res False