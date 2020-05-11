#!/usr/bin/env bash

# On full data
    # 16-32
        # Both acquisitions
            # Greedy policy with self baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_improQR_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--num-target-rows 20 --lr 1e-4 --sample-rate 1 --seed 0 --num-workers 0 --in-chans 1 --lr-gamma 0.1 --num-epochs 30 --lr-step-size 20 --num-pools 4 --pool-stride 1 \
--estimator wr --acq_strat max --acquistion None --center-volume
            # Nongreedy with self baseline (schedule)
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--of-which-four-pools 0 --num-chans 16 --batch-size 4 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
--lr 1e-4 --sample-rate 1 --seed 0 --num-workers 0 --in-chans 1 --num-epochs 50 --num-pools 4 --pool-stride 1 \
--estimator full_step --num-trajectories 8 --num-dev-trajectories 4 --greedy False --data-range volume --baseline-type selfstep \
--scheduler-type multistep --lr-multi-step-size 10 20 --lr-gamma .2 --acquistion None --center-volume --batches-step 4
        # Average oracle
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.run_baseline_models \
--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_cvol_symk/model.pt --recon-model-name nounc \
--batch-size 4 --accelerations 8 --acquisition-steps 16 --sample-rate 1 --seed 0 --num-workers 8 --center-volume --num-epochs 50 --model-type oracle_average \
--acquisition None --wandb

#        # CORPD_FS
#            # Greedy policy with self baseline
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_improQR_model_sweep \
#--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
#--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_symk/model.pt --recon-model-name nounc \
#--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
#--num-target-rows 20 --lr 1e-4 --sample-rate 1 --seed 0 --num-workers 0 --in-chans 1 --lr-gamma 0.1 --num-epochs 50 --lr-step-size 40 --num-pools 4 --pool-stride 1 \
#--estimator wr --acq_strat max --acquistion CORPD_FS --center-volume
#            # Nongreedy with self baseline (schedule)
#CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/var/scratch/tbbakker/anaconda3/envs/fastmri/lib/python3.7/site-packages python -m src.train_RL_model_sweep \
#--dataset fastmri --data-path /var/scratch/tbbakker/data/fastMRI/singlecoil/ --exp-dir /var/scratch/tbbakker/mrimpro/results/ --resolution 128 \
#--recon-model-checkpoint /var/scratch/tbbakker/fastMRI-shi/models/unet/al_nounc_res128_8to4in2_PD_cvol_symk/model.pt --recon-model-name nounc \
#--of-which-four-pools 0 --num-chans 16 --batch-size 16 --impro-model-name convpool --fc-size 256 --accelerations 8 --acquisition-steps 16 --report-interval 100 \
#--lr 1e-4 --sample-rate 1 --seed 0 --num-workers 0 --in-chans 1 --num-epochs 50 --num-pools 4 --pool-stride 1 \
#--estimator full_step --num-trajectories 10 --num-dev-trajectories 10 --greedy False --data-range volume --baseline-type selfstep \
#--scheduler-type multistep --lr-multi-step-size 10 20 30 40 --lr-gamma .5 --acquistion CORPD_FS --center-volume