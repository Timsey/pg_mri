#!/bin/bash

#Set job requirements
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=3
#SBATCH --time 2-12:00:00
#SBATCH --priority=TOP
#SBATCH --job-name=brain
#SBATCH --verbose

echo "Starting..."
echo $HOME
echo $TMPDIR

#Loading modules
source /home/tbbakker/anaconda3/bin/activate fastmri

# Create data dir on scratch
mkdir "$TMPDIR"/data

#Copy input file to scratch
cp -r $HOME/data/fastMRI/brain "$TMPDIR"/data/

#Create output directory on scratch
mkdir "$TMPDIR"/results

nvidia-smi

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.train_improQR_model_sweep \
--dataset fastmri --data-path "$TMPDIR"/data/brain/ --exp-dir "$TMPDIR"/results/ --resolution 256 \
--recon-model-checkpoint /home/tbbakker/Projects/fastMRI-shi/models/unet/al_brain_nonorig_highres256_8to4in2/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 32 --acquisition-steps 28 --report-interval 1000 \
--num-target-rows 8 --lr 5e-5 --sample-rate 0.2 --seed 0 --num-workers 4 --in-chans 1 --lr-gamma 0.1 --num-epochs 50 --num-pools 4 --pool-stride 1 \
--estimator wr --acq_strat sample --acquisition None --center-volume False --lr-step-size 40 --wandb True --do-train-ssim True --num-test-trajectories 1 \
--project mrimpro_brain --original_setting False --low_res False --no_baseline False

#Copy output directory from scratch to home
cp -r "$TMPDIR"/results $HOME/Projects/mrimpro/brain256_results