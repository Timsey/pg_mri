#!/bin/sh

#SBATCH --job-name=nogreedy
#SBATCH --gres=gpu:4  # Hoeveel gpu heb je nodig?
#SBATCH -C GTX980Ti|GTX1080Ti|TitanX  # Welke gpus heb je nodig?

echo "Starting"

source /var/scratch/tbbakker/anaconda3/bin/activate fastmri
nvidia-smi

# On full data
    # 16-32
        # Both acquisitions
            # Nongreedy with self baseline (schedule)
                # 50 epochs
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 4 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 1e-4 --sample-rate 0.5 --seed 0 --num-workers 0 --in-chans 1 --num-epochs 50 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 8 --num-dev-trajectories 4 --greedy False --data-range volume --baseline-type selfstep \
--scheduler-type multistep --lr-multi-step-size 10 20 30 40 --lr-gamma .5 --acquisition None --center-volume True --batches-step 4
                # 30 epochs
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
#--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
#--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
#--of-which-four-pools 0 --num-chans 16 --batch-size 4 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
#--lr 1e-4 --sample-rate 1 --seed 0 --num-workers 0 --in-chans 1 --num-epochs 30 --num-pools 4 --pool-stride 1 \
#--estimator full_step --num-trajectories 8 --num-dev-trajectories 4 --greedy False --data-range volume --baseline-type selfstep \
#--scheduler-type multistep --lr-multi-step-size 10 20 --lr-gamma .2 --acquisition None --center-volume True --batches-step 4