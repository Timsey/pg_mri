#!/bin/sh

#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --priority=TOP
#SBATCH --mem=10G
#SBATCH --verbose
#SBATCH --time 5-0:00:00
#SBATCH --job-name=brain

#SBATCH -D /home/tbbakke/mrimpro

echo "Running..."

source /home/tbbakke/anaconda3/bin/activate fastmri

nvidia-smi

# QR Brain 5 pool fc256 half-channel (hc).
CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.train_improQR_model_sweep \
--dataset fastmri --data-path /home/tbbakke/data/fastMRI/brain/ --exp-dir /home/tbbakke/mrimpro/brain_exp_results/ --resolution 256 \
--recon-model-checkpoint /home/tbbakke/fastMRI-shi/models/unet/al_brain_nonorig_highres256_8to4in2/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 8 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 1000 \
--num-target-rows 8 --lr 5e-5 --sample-rate 0.2 --seed 0 --num-workers 4 --in-chans 1 --lr-gamma 0.1 --num-epochs 50 --num-pools 5 --pool-stride 1 \
--estimator wr --acq_strat sample --acquisition None --center-volume False --lr-step-size 40 --wandb True --do-train-ssim False --num-test-trajectories 1 \
--project mrimpro_brain --original_setting False --low_res False --no_baseline False

# Long
CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.train_improQR_model_sweep \
--dataset fastmri --data-path /home/tbbakke/data/fastMRI/brain/ --exp-dir /home/tbbakke/mrimpro/brain_exp_results/ --resolution 256 \
--recon-model-checkpoint /home/tbbakke/fastMRI-shi/models/unet/al_brain_nonorig_highres256_8to4in2/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 8 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 32 --acquisition-steps 28 --report-interval 1000 \
--num-target-rows 8 --lr 5e-5 --sample-rate 0.2 --seed 0 --num-workers 4 --in-chans 1 --lr-gamma 0.1 --num-epochs 50 --num-pools 5 --pool-stride 1 \
--estimator wr --acq_strat sample --acquisition None --center-volume False --lr-step-size 40 --wandb True --do-train-ssim True --num-test-trajectories 1 \
--project mrimpro_brain --original_setting False --low_res False --no_baseline False