#!/usr/bin/env bash

source /home/timsey/anaconda3/bin/activate rim

echo "---------------------------------"

#CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.train_improQR_model_sweep \
#--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp-dir /home/timsey/Projects/mrimpro/var_results/QR/ --resolution 128 \
#--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
#--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
#--num-target-rows 8 --lr 1e-4 --sample-rate 0.1 --seed 0 --num-workers 4 --in-chans 1 --lr-gamma 0.1 --num-epochs 30 --num-pools 4 --pool-stride 1 \
#--estimator wr --acq_strat max --acquisition None --center-volume True --scheduler-type multistep --lr-multi-step-size 10 20 \
#--wandb True --do-train-ssim True

echo "---------------------------------"

#CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.train_improQR_model_sweep \
#--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp-dir /home/timsey/Projects/mrimpro/exp_results/ --resolution 128 \
#--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
#--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 1000 \
#--num-target-rows 8 --lr 5e-5 --sample-rate 0.5 --seed 0 --num-workers 4 --in-chans 1 --lr-gamma 0.1 --num-epochs 50 --num-pools 4 --pool-stride 1 \
#--estimator wr --acq_strat sample --acquisition None --center-volume True --lr-step-size 40 \
#--wandb True --do-train-ssim True --num-test-trajectories 1


#CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.train_improQR_model_sweep \
#--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp-dir /home/timsey/Projects/mrimpro/exp_results/ --resolution 128 \
#--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
#--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 32 --acquisition-steps 28 --report-interval 1000 \
#--num-target-rows 8 --lr 5e-5 --sample-rate 0.5 --seed 0 --num-workers 4 --in-chans 1 --lr-gamma 0.1 --num-epochs 50 --num-pools 4 --pool-stride 1 \
#--estimator wr --acq_strat sample --acquisition None --center-volume True --lr-step-size 40 \
#--wandb True --do-train-ssim True --num-test-trajectories 1

# wor
CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.train_improQR_model_sweep \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp-dir /home/timsey/Projects/mrimpro/gamma_results_fix/ --resolution 128 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 1000 \
--num-target-rows 8 --lr 5e-5 --sample-rate 0.5 --seed 0 --num-workers 4 --in-chans 1 --lr-gamma 0.1 --num-epochs 50 --num-pools 4 --pool-stride 1 \
--estimator wor --acq_strat sample --acquisition None --center-volume True --lr-step-size 40 --project mrimpro_gamma_fix \
--wandb True --do-train-ssim True --num-test-trajectories 1 --original_setting True --low_res False --no_baseline False

# Low lr
CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.train_improQR_model_sweep \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp-dir /home/timsey/Projects/mrimpro/gamma_results_fix/ --resolution 128 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 1000 \
--num-target-rows 8 --lr 1e-5 --sample-rate 0.5 --seed 0 --num-workers 4 --in-chans 1 --lr-gamma 0.1 --num-epochs 50 --num-pools 4 --pool-stride 1 \
--estimator wor --acq_strat sample --acquisition None --center-volume True --lr-step-size 40 --project mrimpro_gamma_fix \
--wandb True --do-train-ssim True --num-test-trajectories 1 --original_setting True --low_res False --no_baseline False

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.train_improQR_model_sweep \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp-dir /home/timsey/Projects/mrimpro/gamma_results_fix/ --resolution 128 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 1000 \
--num-target-rows 8 --lr 2e-5 --sample-rate 0.5 --seed 0 --num-workers 4 --in-chans 1 --lr-gamma 0.1 --num-epochs 50 --num-pools 4 --pool-stride 1 \
--estimator wor --acq_strat sample --acquisition None --center-volume True --lr-step-size 40 --project mrimpro_gamma_fix \
--wandb True --do-train-ssim True --num-test-trajectories 1 --original_setting True --low_res False --no_baseline False

# No baseline
CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.train_improQR_model_sweep \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp-dir /home/timsey/Projects/mrimpro/gamma_results_fix/ --resolution 128 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 1000 \
--num-target-rows 8 --lr 5e-5 --sample-rate 0.5 --seed 0 --num-workers 4 --in-chans 1 --lr-gamma 0.1 --num-epochs 50 --num-pools 4 --pool-stride 1 \
--estimator wr --acq_strat sample --acquisition None --center-volume True --lr-step-size 40 --project mrimpro_gamma_fix \
--wandb True --do-train-ssim True --num-test-trajectories 1 --original_setting True --low_res False --no_baseline True

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.train_improQR_model_sweep \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp-dir /home/timsey/Projects/mrimpro/gamma_results_fix/ --resolution 128 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 1000 \
--num-target-rows 8 --lr 1e-5 --sample-rate 0.5 --seed 0 --num-workers 4 --in-chans 1 --lr-gamma 0.1 --num-epochs 50 --num-pools 4 --pool-stride 1 \
--estimator wr --acq_strat sample --acquisition None --center-volume True --lr-step-size 40 --project mrimpro_gamma_fix \
--wandb True --do-train-ssim True --num-test-trajectories 1 --original_setting True --low_res False --no_baseline True

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.train_improQR_model_sweep \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp-dir /home/timsey/Projects/mrimpro/gamma_results_fix/ --resolution 128 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 1000 \
--num-target-rows 8 --lr 2e-5 --sample-rate 0.5 --seed 0 --num-workers 4 --in-chans 1 --lr-gamma 0.1 --num-epochs 50 --num-pools 4 --pool-stride 1 \
--estimator wr --acq_strat sample --acquisition None --center-volume True --lr-step-size 40 --project mrimpro_gamma_fix \
--wandb True --do-train-ssim True --num-test-trajectories 1 --original_setting True --low_res False --no_baseline True

# Another base wor
CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.train_improQR_model_sweep \
--dataset fastmri --data-path /home/timsey/HDD/data/fastMRI/singlecoil/ --exp-dir /home/timsey/Projects/mrimpro/gamma_results_fix/ --resolution 128 \
--recon-model-checkpoint /home/timsey/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 1000 \
--num-target-rows 8 --lr 5e-5 --sample-rate 0.5 --seed 0 --num-workers 4 --in-chans 1 --lr-gamma 0.1 --num-epochs 50 --num-pools 4 --pool-stride 1 \
--estimator wor --acq_strat sample --acquisition None --center-volume True --lr-step-size 40 --project mrimpro_gamma_fix \
--wandb True --do-train-ssim True --num-test-trajectories 1 --original_setting True --low_res False --no_baseline False