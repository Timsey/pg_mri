#!/bin/sh

#SBATCH --job-name=nogreedy
#SBATCH --gres=gpu:1  # Hoeveel gpu heb je nodig?
#SBATCH -C GTX980Ti|GTX1080Ti|TitanX  # Welke gpus heb je nodig?

echo "Starting"

source /var/scratch/tbbakker/anaconda3/bin/activate fastmri
nvidia-smi

# On half data (need data-state!)
    # 16-32
        # Both acquisitions
            # Greedy policy with self baseline
                # 30 epochs
CUDA_VISIBLE_DEVICES=0 HDF5_USE_FILE_LOCKING=FALSE PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_improQR_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/exp_results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--num-target-rows 8 --lr 1e-4 --sample-rate 0.5 --seed 0 --num-workers 4 --in-chans 1 --lr-gamma 0.1 --num-epochs 30 --num-pools 4 --pool-stride 1 \
--estimator wr --acq_strat max --acquisition None --center-volume True --use-data-state True --scheduler-type multistep --lr-multi-step-size 10 20