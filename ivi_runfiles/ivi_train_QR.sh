#!/bin/sh

#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --priority=TOP
#SBATCH --mem=10G
#SBATCH --verbose
#SBATCH --time 1-0:00:00
#SBATCH --job-name=greedy

#SBATCH -D /home/tbbakke/alphamri

echo "Running..."

source /home/tbbakke/anaconda3/bin/activate ml

nvidia-smi

# Do your stuff

CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE python -m src.train_improQR_model_sweep \
--dataset fastmri --data-path /home/tbbakke/data/fastMRI/singlecoil/ --exp-dir /home/tbbakke/mrimpro/exp_results/ --resolution 128 \
--recon-model-checkpoint /home/tbbakke/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--num-target-rows 8 --lr 1e-4 --sample-rate 0.5 --seed 0 --num-workers 4 --in-chans 1 --lr-gamma 0.1 --num-epochs 30 --num-pools 4 --pool-stride 1 \
--estimator wr --acq_strat max --acquisition None --center-volume True --use-data-state True --scheduler-type multistep --lr-multi-step-size 10 20