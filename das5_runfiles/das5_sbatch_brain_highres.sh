#!/bin/sh

#SBATCH --job-name=nogreedy
#SBATCH --gres=gpu:4  # Hoeveel gpu heb je nodig?
#SBATCH -C GTX980Ti|GTX1080Ti|TitanX  # Welke gpus heb je nodig?

echo "Starting"

source /var/scratch/tbbakker/anaconda3/bin/activate fastmri
nvidia-smi

CUDA_VISIBLE_DEVICES=0,1,2,3 HDF5_USE_FILE_LOCKING=FALSE PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_improQR_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/brain/ --exp-dir /var/scratch/tbbakker/mrimpro/brain_exp_results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_brain_nonorig_highres128_8to4in2/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 1000 \
--num-target-rows 8 --lr 5e-5 --sample-rate 0.2 --seed 0 --num-workers 4 --in-chans 1 --lr-gamma 0.1 --num-epochs 50 --num-pools 4 --pool-stride 1 \
--estimator wr --acq_strat sample --acquisition None --center-volume False --lr-step-size 40 --wandb True --do-train-ssim True --num-test-trajectories 1 \
--project mrimpro_brain --original_setting False --low_res False

CUDA_VISIBLE_DEVICES=0,1,2,3 HDF5_USE_FILE_LOCKING=FALSE PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/brain/ --exp-dir /var/scratch/tbbakker/mrimpro/brain_exp_results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_brain_nonorig_highres128_8to4in2/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 4 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 1000 \
--lr 5e-5 --sample-rate 0.2 --seed 0 --num-workers 4 --in-chans 1 --num-epochs 50 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 8 --num-dev-trajectories 4 --greedy False --data-range volume --baseline-type selfstep \
--scheduler-type multistep --lr-multi-step-size 10 20 30 40 --lr-gamma .5 --acquisition None --center-volume False --batches-step 4 \
--wandb True --do-train-ssim True --project mrimpro_brain --original_setting False --low_res False --gamma 1.0

