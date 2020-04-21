#!/bin/sh

#SBATCH --job-name=nogreedy
#SBATCH --gres=gpu:4  # Hoeveel gpu heb je nodig?
#SBATCH -C GTX980Ti|TitanX  # Welke gpus heb je nodig?

echo "Starting"

source /var/scratch/tbbakker/anaconda3/bin/activate fastmri
nvidia-smi

# Nongreedy with statestep baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 1e-4 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --lr-gamma 0.1 --num-epochs 100 --lr-step-size 80 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 4 --num-dev-trajectories 4 --greedy False --data-range gt --baseline-type statestep

# Nongreedy with statestep baseline, lr 1e-5
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 1e-5 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --lr-gamma 0.1 --num-epochs 100 --lr-step-size 80 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 4 --num-dev-trajectories 4 --greedy False --data-range gt --baseline-type statestep

# Greedy with statestep baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 1e-4 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --lr-gamma 0.1 --num-epochs 100 --lr-step-size 80 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 4 --num-dev-trajectories 4 --greedy True --data-range gt --baseline-type statestep