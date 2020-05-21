#!/bin/bash

#Set job requirements
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=6
#SBATCH --time 5-0:00:00
#SBATCH --priority=TOP
#SBATCH --job-name=nongreedy
#SBATCH --verbose

echo "Starting..."
echo $HOME
echo $TMPDIR

#Loading modules
source /home/tbbakker/anaconda3/bin/activate fastmri

# Create data dir on scratch
mkdir "$TMPDIR"/data

#Copy input file to scratch
cp -r $HOME/data/fastMRI/singlecoil "$TMPDIR"/data/

#Create output directory on scratch
mkdir "$TMPDIR"/results

nvidia-smi

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.train_RL_model_sweep \
--dataset fastmri --data-path "$TMPDIR"/data/singlecoil/ --exp-dir "$TMPDIR"/results/ --resolution 128 \
--recon-model-checkpoint /home/tbbakker/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 4 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 1e-4 --sample-rate 0.5 --seed 0 --num-workers 4 --in-chans 1 --num-epochs 50 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 8 --num-dev-trajectories 8 --greedy False --data-range volume --baseline-type selfstep \
--scheduler-type multistep --lr-multi-step-size 10 20 30 40 --lr-gamma .5 --acquisition None --center-volume True --batches-step 4 \
--wandb True --do-train-ssim True

#Copy output directory from scratch to home
cp -r "$TMPDIR"/results $HOME/Projects/mrimpro/results