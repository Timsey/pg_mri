#!/bin/bash

#Set job requirements
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time 0-1:00:00
#SBATCH --priority=TOP
#SBATCH --job-name=greedy
#SBATCH --verbose

echo "Starting..."
echo $HOME
echo $TMPDIR

#Loading modules
source /home/tbbakker/anaconda3/bin/activate fastmri

nvidia-smi

# Create data dir on scratch
mkdir "$TMPDIR"/data

#Copy input file to scratch
cp -r $HOME/data/fastMRI/singlecoil "$TMPDIR"/data/

#Create output directory on scratch
mkdir "$TMPDIR"/results

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.train_improQR_model_sweep \
--dataset fastmri --data-path "$TMPDIR"/data/singlecoil/ --exp-dir "$TMPDIR"/results/ --resolution 128 \
--recon-model-checkpoint /home/tbbakker/Projects/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 8 --batch-size 64 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 2 --report-interval 100 \
--num-target-rows 2 --lr 1e-4 --sample-rate 0.04 --seed 0 --num-workers 4 --in-chans 1 --lr-gamma 0.1 --num-epochs 1 --num-pools 4 --pool-stride 1 \
--estimator wr --acq_strat sample --acquisition None --center-volume True \
--wandb True --do-train-ssim False

#Copy output directory from scratch to home
cp -r "$TMPDIR"/results $HOME/Projects/mrimpro/results