#!/bin/sh

#SBATCH --job-name=nogreedy
#SBATCH --gres=gpu:4  # Hoeveel gpu heb je nodig?
#SBATCH -C GTX980Ti|GTX1080Ti|TitanX  # Welke gpus heb je nodig?

echo "Starting"

source /var/scratch/tbbakker/anaconda3/bin/activate fastmri
nvidia-smi

# Nongreedy models with in-between lrs (compare 776, 778, 783)
# Step baseline
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
#--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
#--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
#--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
#--lr 3e-5 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --lr-gamma 0.1 --num-epochs 50 --lr-step-size 40 --num-pools 4 --pool-stride 1 \
#--estimator full_step --num-trajectories 4 --num-dev-trajectories 10 --greedy False --data-range gt --baseline-type step
## Statestep baseline
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
#--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
#--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
#--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
#--lr 3e-5 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --lr-gamma 0.1 --num-epochs 50 --lr-step-size 40 --num-pools 4 --pool-stride 1 \
#--estimator full_step --num-trajectories 4 --num-dev-trajectories 10 --greedy False --data-range gt --baseline-type statestep
# Step baseline, lr schedule
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
#--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
#--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
#--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
#--lr 1e-4 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --num-epochs 50 --num-pools 4 --pool-stride 1 \
#--estimator full_step --num-trajectories 4 --num-dev-trajectories 10 --greedy False --data-range gt --baseline-type step \
#--scheduler-type multistep --lr-multi-step-size 10 20 30 40 --lr-gamma .5

## Redoing baselines for volume SSIM
## Nongreedy with state baseline, trj4
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
#--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
#--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
#--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
#--lr 1e-4 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --lr-gamma 0.1 --num-epochs 50 --lr-step-size 40 --num-pools 4 --pool-stride 1 \
#--estimator full_step --num-trajectories 4 --num-dev-trajectories 10 --greedy False --data-range volume --baseline-type step
## Nongreedy with state baseline, trj2
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
#--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
#--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
#--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
#--lr 1e-4 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --lr-gamma 0.1 --num-epochs 50 --lr-step-size 40 --num-pools 4 --pool-stride 1 \
#--estimator full_step --num-trajectories 2 --num-dev-trajectories 10 --greedy False --data-range volume --baseline-type step
## Greedy with state baseline, trj10
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
#--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
#--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
#--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
#--lr 1e-4 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --lr-gamma 0.1 --num-epochs 50 --lr-step-size 40 --num-pools 4 --pool-stride 1 \
#--estimator full_step --num-trajectories 10 --num-dev-trajectories 10 --greedy True --data-range volume --baseline-type step
## Greedy with state baseline, trj4
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
#--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
#--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
#--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
#--lr 1e-4 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --lr-gamma 0.1 --num-epochs 50 --lr-step-size 40 --num-pools 4 --pool-stride 1 \
#--estimator full_step --num-trajectories 4 --num-dev-trajectories 10 --greedy True --data-range volume --baseline-type step
## Greedy with state baseline, trj2
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
#--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
#--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
#--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
#--lr 1e-4 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --lr-gamma 0.1 --num-epochs 50 --lr-step-size 40 --num-pools 4 --pool-stride 1 \
#--estimator full_step --num-trajectories 2 --num-dev-trajectories 10 --greedy True --data-range volume --baseline-type step

# Volume SSIM non-greedy with in-between lr
# Nongreedy with state baseline, trj4
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
#--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
#--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
#--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
#--lr 3e-5 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --lr-gamma 0.1 --num-epochs 50 --lr-step-size 40 --num-pools 4 --pool-stride 1 \
#--estimator full_step --num-trajectories 4 --num-dev-trajectories 10 --greedy False --data-range volume --baseline-type step
## Nongreedy with state baseline, trj2
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
#--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
#--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
#--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
#--lr 3e-5 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --lr-gamma 0.1 --num-epochs 50 --lr-step-size 40 --num-pools 4 --pool-stride 1 \
#--estimator full_step --num-trajectories 2 --num-dev-trajectories 10 --greedy False --data-range volume --baseline-type step
# Step baseline, lr schedule, trj4
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
#--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
#--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
#--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
#--lr 1e-4 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --num-epochs 50 --num-pools 4 --pool-stride 1 \
#--estimator full_step --num-trajectories 4 --num-dev-trajectories 10 --greedy False --data-range volume --baseline-type step \
#--scheduler-type multistep --lr-multi-step-size 10 20 30 40 --lr-gamma .5


# Greedy (train tr4, dev tr10)
    # Short
        # High lr
            # Step baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 1e-4 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --lr-gamma 0.1 --num-epochs 50 --lr-step-size 40 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 4 --num-dev-trajectories 10 --greedy True --data-range volume --baseline-type step
            # Statestep baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 1e-4 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --lr-gamma 0.1 --num-epochs 50 --lr-step-size 40 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 4 --num-dev-trajectories 10 --greedy True --data-range volume --baseline-type statestep
        # Mid lr
            # Step baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 3e-5 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --lr-gamma 0.1 --num-epochs 50 --lr-step-size 40 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 4 --num-dev-trajectories 10 --greedy True --data-range volume --baseline-type step
            # Statestep baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 3e-5 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --lr-gamma 0.1 --num-epochs 50 --lr-step-size 40 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 4 --num-dev-trajectories 10 --greedy True --data-range volume --baseline-type statestep
        # Low lr
            # Step baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 1e-5 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --lr-gamma 0.1 --num-epochs 50 --lr-step-size 40 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 4 --num-dev-trajectories 10 --greedy True --data-range volume --baseline-type step
            # Statestep baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 1e-5 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --lr-gamma 0.1 --num-epochs 50 --lr-step-size 40 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 4 --num-dev-trajectories 10 --greedy True --data-range volume --baseline-type statestep
        # Schedule
            # Step baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 1e-4 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --num-epochs 50 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 4 --num-dev-trajectories 10 --greedy True --data-range volume --baseline-type step \
--scheduler-type multistep --lr-multi-step-size 10 20 30 40 --lr-gamma .5
            # Statestep baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 1e-4 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --num-epochs 50 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 4 --num-dev-trajectories 10 --greedy True --data-range volume --baseline-type statestep \
--scheduler-type multistep --lr-multi-step-size 10 20 30 40 --lr-gamma .5
    # Long
        # High lr
            # Step baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 1e-4 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --lr-gamma 0.1 --num-epochs 100 --lr-step-size 40 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 4 --num-dev-trajectories 10 --greedy True --data-range volume --baseline-type step
            # Statestep baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 1e-4 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --lr-gamma 0.1 --num-epochs 100 --lr-step-size 40 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 4 --num-dev-trajectories 10 --greedy True --data-range volume --baseline-type statestep
        # Mid lr
            # Step baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 3e-5 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --lr-gamma 0.1 --num-epochs 100 --lr-step-size 40 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 4 --num-dev-trajectories 10 --greedy True --data-range volume --baseline-type step
            # Statestep baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 3e-5 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --lr-gamma 0.1 --num-epochs 100 --lr-step-size 40 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 4 --num-dev-trajectories 10 --greedy True --data-range volume --baseline-type statestep
        # Low lr
            # Step baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 1e-5 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --lr-gamma 0.1 --num-epochs 100 --lr-step-size 40 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 4 --num-dev-trajectories 10 --greedy True --data-range volume --baseline-type step
            # Statestep baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 1e-5 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --lr-gamma 0.1 --num-epochs 100 --lr-step-size 40 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 4 --num-dev-trajectories 10 --greedy True --data-range volume --baseline-type statestep
        # Schedule
            # Step baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 1e-4 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --num-epochs 100 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 4 --num-dev-trajectories 10 --greedy True --data-range volume --baseline-type step \
--scheduler-type multistep --lr-multi-step-size 10 30 60 90 --lr-gamma .5
            # Statestep baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 1e-4 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --num-epochs 100 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 4 --num-dev-trajectories 10 --greedy True --data-range volume --baseline-type statestep \
--scheduler-type multistep --lr-multi-step-size 10 30 60 90 --lr-gamma .5



# Nongreedy (train tr4, dev tr10)
    # Short
        # High lr
            # Step baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 1e-4 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --lr-gamma 0.1 --num-epochs 50 --lr-step-size 40 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 4 --num-dev-trajectories 10 --greedy False --data-range volume --baseline-type step
            # Statestep baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 1e-4 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --lr-gamma 0.1 --num-epochs 50 --lr-step-size 40 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 4 --num-dev-trajectories 10 --greedy False --data-range volume --baseline-type statestep
        # Mid lr
            # Step baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 3e-5 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --lr-gamma 0.1 --num-epochs 50 --lr-step-size 40 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 4 --num-dev-trajectories 10 --greedy False --data-range volume --baseline-type step
            # Statestep baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 3e-5 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --lr-gamma 0.1 --num-epochs 50 --lr-step-size 40 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 4 --num-dev-trajectories 10 --greedy False --data-range volume --baseline-type statestep
        # Low lr
            # Step baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 1e-5 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --lr-gamma 0.1 --num-epochs 50 --lr-step-size 40 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 4 --num-dev-trajectories 10 --greedy False --data-range volume --baseline-type step
            # Statestep baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 1e-5 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --lr-gamma 0.1 --num-epochs 50 --lr-step-size 40 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 4 --num-dev-trajectories 10 --greedy False --data-range volume --baseline-type statestep
        # Schedule
            # Step baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 1e-4 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --num-epochs 50 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 4 --num-dev-trajectories 10 --greedy False --data-range volume --baseline-type step \
--scheduler-type multistep --lr-multi-step-size 10 20 30 40 --lr-gamma .5
            # Statestep baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 1e-4 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --num-epochs 50 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 4 --num-dev-trajectories 10 --greedy False --data-range volume --baseline-type statestep \
--scheduler-type multistep --lr-multi-step-size 10 20 30 40 --lr-gamma .5
    # Long
        # High lr
            # Step baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 1e-4 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --lr-gamma 0.1 --num-epochs 100 --lr-step-size 40 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 4 --num-dev-trajectories 10 --greedy False --data-range volume --baseline-type step
            # Statestep baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 1e-4 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --lr-gamma 0.1 --num-epochs 100 --lr-step-size 40 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 4 --num-dev-trajectories 10 --greedy False --data-range volume --baseline-type statestep
        # Mid lr
            # Step baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 3e-5 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --lr-gamma 0.1 --num-epochs 100 --lr-step-size 40 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 4 --num-dev-trajectories 10 --greedy False --data-range volume --baseline-type step
            # Statestep baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 3e-5 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --lr-gamma 0.1 --num-epochs 100 --lr-step-size 40 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 4 --num-dev-trajectories 10 --greedy False --data-range volume --baseline-type statestep
        # Low lr
            # Step baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 1e-5 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --lr-gamma 0.1 --num-epochs 100 --lr-step-size 40 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 4 --num-dev-trajectories 10 --greedy False --data-range volume --baseline-type step
            # Statestep baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 1e-5 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --lr-gamma 0.1 --num-epochs 100 --lr-step-size 40 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 4 --num-dev-trajectories 10 --greedy False --data-range volume --baseline-type statestep
        # Schedule
            # Step baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 1e-4 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --num-epochs 100 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 4 --num-dev-trajectories 10 --greedy False --data-range volume --baseline-type step \
--scheduler-type multistep --lr-multi-step-size 10 30 60 90 --lr-gamma .5
            # Statestep baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_ch16_b64_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 1e-4 --sample-rate 0.04 --seed 42 --num-workers 0 --in-chans 1 --num-epochs 100 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 4 --num-dev-trajectories 10 --greedy False --data-range volume --baseline-type statestep \
--scheduler-type multistep --lr-multi-step-size 10 30 60 90 --lr-gamma .5