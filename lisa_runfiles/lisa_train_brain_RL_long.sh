#!/bin/bash

#Set job requirements
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=24
#SBATCH --time 5-00:00:00
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

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.train_RL_model_sweep \
--dataset fastmri --data-path "$TMPDIR"/data/brain/ --exp-dir "$TMPDIR"/results/ --resolution 256 \
--recon-model-checkpoint /home/tbbakker/Projects/fastMRI-shi/models/unet/al_brain_nonorig_highres256_8to4in2/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 8 --batch-size 4 --impro-model-name convpool --fc-size 256 --accelerations 32 --acquisition-steps 28 --report-interval 1000 \
--lr 5e-5 --sample-rate 0.2 --seed 0 --num-workers 4 --in-chans 1 --num-epochs 50 --num-pools 5 --pool-stride 1 \
--estimator full_step --num-trajectories 8 --num-dev-trajectories 4 --greedy False --data-range volume --baseline-type selfstep \
--scheduler-type multistep --lr-multi-step-size 10 20 30 40 --lr-gamma .5 --acquisition None --center-volume False --batches-step 4 \
--wandb True --do-train-ssim False --project mrimpro_brain --original_setting False --low_res False --gamma 1.0