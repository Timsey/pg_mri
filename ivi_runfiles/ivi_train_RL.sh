#!/bin/sh

#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --priority=TOP
#SBATCH --mem=20G
#SBATCH --verbose
#SBATCH --time 7-0:00:00
#SBATCH --job-name=nogreedy

#SBATCH -D /home/tbbakke/mrimpro

echo "Running..."

source /home/tbbakke/anaconda3/bin/activate ml

nvidia-smi

# Do your stuff

CUDA_VISIBLE_DEVICES=0,1 HDF5_USE_FILE_LOCKING=FALSE python -m src.train_improQR_model_sweep \
--dataset fastmri --data-path /home/tbbakke/data/fastMRI/singlecoil/ --exp-dir /home/tbbakke/mrimpro/exp_results/ --resolution 128 \
--recon-model-checkpoint /home/tbbakke/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 4 --impro-model-name convpool --fc-size 256 --accelerations 32 --acquisition-steps 28 --report-interval 100 \
--lr 1e-4 --sample-rate 0.5 --seed 0 --num-workers 4 --in-chans 1 --num-epochs 50 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 8 --num-dev-trajectories 8 --greedy False --data-range volume --baseline-type selfstep \
--scheduler-type multistep --lr-multi-step-size 10 20 30 40 --lr-gamma .5 --acquisition None --center-volume True --batches-step 4 \
--wandb True --do-train-ssim True

CUDA_VISIBLE_DEVICES=0,1 HDF5_USE_FILE_LOCKING=FALSE python -m src.train_improQR_model_sweep \
--dataset fastmri --data-path /home/tbbakke/data/fastMRI/singlecoil/ --exp-dir /home/tbbakke/mrimpro/exp_results/ --resolution 128 \
--recon-model-checkpoint /home/tbbakke/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 4 --impro-model-name convpool --fc-size 256 --accelerations 32 --acquisition-steps 28 --report-interval 100 \
--lr 5e-5 --sample-rate 0.5 --seed 0 --num-workers 4 --in-chans 1 --num-epochs 50 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 8 --num-dev-trajectories 8 --greedy False --data-range volume --baseline-type selfstep \
--scheduler-type multistep --lr-multi-step-size 10 20 30 40 --lr-gamma .5 --acquisition None --center-volume True --batches-step 4 \
--wandb True --do-train-ssim True