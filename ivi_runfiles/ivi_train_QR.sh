#!/bin/sh

#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --priority=TOP
#SBATCH --mem=10G
#SBATCH --verbose
#SBATCH --time 5-0:00:00
#SBATCH --job-name=greedy

#SBATCH -D /home/tbbakke/mrimpro

echo "Running..."

source /home/tbbakke/anaconda3/bin/activate fastmri

nvidia-smi

# Do your stuff

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.train_improQR_model_sweep \
--dataset fastmri --data-path /home/tbbakke/data/fastMRI/singlecoil/ --exp-dir /home/tbbakke/mrimpro/gamma_results_fix/ --resolution 128 \
--recon-model-checkpoint /home/tbbakke/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 1000 \
--num-target-rows 8 --lr 5e-5 --sample-rate 0.5 --seed 0 --num-workers 4 --in-chans 1 --lr-gamma 0.1 --num-epochs 50 --num-pools 4 --pool-stride 1 \
--estimator wor --acq_strat sample --acquisition None --center-volume True --lr-step-size 40 --project mrimpro_gamma_fix \
--wandb True --do-train-ssim False --num-test-trajectories 1 --original_setting True --low_res False --no_baseline False

#CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.train_improQR_model_sweep \
#--dataset fastmri --data-path /home/tbbakke/data/fastMRI/singlecoil/ --exp-dir /home/tbbakke/mrimpro/gamma_results_fix/ --resolution 128 \
#--recon-model-checkpoint /home/tbbakke/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
#--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 32 --acquisition-steps 28 --report-interval 1000 \
#--num-target-rows 8 --lr 5e-5 --sample-rate 0.5 --seed 0 --num-workers 4 --in-chans 1 --lr-gamma 0.1 --num-epochs 50 --num-pools 4 --pool-stride 1 \
#--estimator wor --acq_strat sample --acquisition None --center-volume True --lr-step-size 40 --project mrimpro_gamma_fix \
#--wandb True --do-train-ssim False --num-test-trajectories 1 --original_setting True --low_res False --no_baseline False
#
#CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.train_improQR_model_sweep \
#--dataset fastmri --data-path /home/tbbakke/data/fastMRI/singlecoil/ --exp-dir /home/tbbakke/mrimpro/gamma_results_fix/ --resolution 128 \
#--recon-model-checkpoint /home/tbbakke/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
#--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 32 --acquisition-steps 28 --report-interval 1000 \
#--num-target-rows 8 --lr 5e-5 --sample-rate 0.5 --seed 0 --num-workers 4 --in-chans 1 --lr-gamma 0.1 --num-epochs 50 --num-pools 4 --pool-stride 1 \
#--estimator wor --acq_strat sample --acquisition None --center-volume True --lr-step-size 40 --project mrimpro_gamma_fix \
#--wandb True --do-train-ssim False --num-test-trajectories 1 --original_setting True --low_res False --no_baseline False
#
#CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.train_improQR_model_sweep \
#--dataset fastmri --data-path /home/tbbakke/data/fastMRI/singlecoil/ --exp-dir /home/tbbakke/mrimpro/gamma_results_fix/ --resolution 128 \
#--recon-model-checkpoint /home/tbbakke/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
#--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 32 --acquisition-steps 28 --report-interval 1000 \
#--num-target-rows 8 --lr 5e-5 --sample-rate 0.5 --seed 0 --num-workers 4 --in-chans 1 --lr-gamma 0.1 --num-epochs 50 --num-pools 4 --pool-stride 1 \
#--estimator wor --acq_strat sample --acquisition None --center-volume True --lr-step-size 40 --project mrimpro_gamma_fix \
#--wandb True --do-train-ssim False --num-test-trajectories 1 --original_setting True --low_res False --no_baseline False
#
#CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.train_improQR_model_sweep \
#--dataset fastmri --data-path /home/tbbakke/data/fastMRI/singlecoil/ --exp-dir /home/tbbakke/mrimpro/gamma_results_fix/ --resolution 128 \
#--recon-model-checkpoint /home/tbbakke/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
#--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 32 --acquisition-steps 28 --report-interval 1000 \
#--num-target-rows 8 --lr 5e-5 --sample-rate 0.5 --seed 0 --num-workers 4 --in-chans 1 --lr-gamma 0.1 --num-epochs 50 --num-pools 4 --pool-stride 1 \
#--estimator wor --acq_strat sample --acquisition None --center-volume True --lr-step-size 40 --project mrimpro_gamma_fix \
#--wandb True --do-train-ssim False --num-test-trajectories 1 --original_setting True --low_res False --no_baseline False