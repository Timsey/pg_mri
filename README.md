# Experimental design for MRI by greedy policy search
This repository is the official implementation of: [Experimental design for MRI by greedy policy search]() (NeurIPS, 2020).

![](https://user-images.githubusercontent.com/35295146/96741535-30e79a80-13c2-11eb-8785-5263a3b522d0.png)

## Requirements

To install requirements:

```
pip install -r requirements.txt
```

Or using conda:

```
conda env create -f environment.yml
```

Note: these environments are complete, but not minimal.


Data should be stored as follows, as the top-level folders are hardcoded into our data loaders (see `src/helpers/data_loading.py`).
```
<path_to_data>
  singlecoil_train/
  singlecoil_val/
  singlecoil_test/
```

This corresponds somewhat to the default download settings of the fastMRI data repository. For both Knee and Brain datasets the original test data contains not ground truths, and so we construct a new `singlecoil_test` from `singlecoil_train`, as explained in the paper. The IPython notebook `split_data.ipynb` in `notebooks` provides a utility for this.
The default Brain data directory names are `multicoil_` instead of `singlecoil_`. Note that we do use not use the multicoil k-space and instead construct singlecoil k-space from the ground truth images. To save on I/O, we recommend removing the multicoil k-space from the `.h5` files (`split_data.ipynb` contains a utility for this). For naming consistency, we have also renamed `multicoil_` to `singlecoil_` for Brain data.


## Training
All commands should be run from the repository root folder. Logging is done using Tensorboard.
### Reconstruction models
Scripts will store results directly in <path_to_output>, so make sure to change this for different runs!
#### Knee
```
CUDA_VISIBLE_DEVICES=0 python -m src.train_reconstruction --data_path <path_to_data> --exp_dir <path_to_output>
```
#### Brain
```
CUDA_VISIBLE_DEVICES=0 python -m src.train_reconstruction --dataset brain --data_path <path_to_data> --exp_dir <path_to_output> --resolution 256 --center_volume False
```

### Policy models
Training is done using the train_policy.py script. Logging is done with Tensorboard and <cite>[Weights and Biases][1]</cite> (see the `wandb` argument). Note that the `wandb` argument is optional (set `wandb=False` to forego usage), but some of the visualisation notebooks require stored wandb runs to function.
Note: effective train batch size is given as batch_size * batches_step. Higher batches step results in slower training, but less memory used (this is mostly relevant for non-greedy models). If more GPUs are available, batch size can be increase (and batches_step reduced).
Scripts will create a datetime stamped folder in <path_to_output> to store all results in.

Note that we do not include code for reproducing the AlphaZero results, as the original repository is not ours and has not been open sourced.

#### Knee
##### Base horizon greedy (1GPU)
```
CUDA_VISIBLE_DEVICES=0 python -m src.train_policy --data_path <path_to_data> --exp_dir <path_to_output> --recon_model_checkpoint <path_to_reconstruction_model.pt> --model_type greedy --project <wandb_project_name> --wandb True
```
##### Base horizon non-greedy (1GPU)
```
CUDA_VISIBLE_DEVICES=0 python -m src.train_policy --data_path <path_to_data> --exp_dir <path_to_output> --recon_model_checkpoint <path_to_reconstruction_model.pt> --model_type nongreedy --batch_size 4 --batches_step 4 --gamma 1 --lr_gamma 0.5 --scheduler_type multistep --project <wandb_project_name> --wandb True
```
##### Long horizon greedy (1GPU)
```
CUDA_VISIBLE_DEVICES=0 python -m src.train_policy --data_path <path_to_data> --exp_dir <path_to_output> --recon_model_checkpoint <path_to_reconstruction_model.pt> --model_type greedy --accelerations 32 --acquisition_steps 28 --project <wandb_project_name> --wandb True
```
##### Long horizon non-greedy (2GPU)
```
CUDA_VISIBLE_DEVICES=0,1 python -m src.train_policy --data_path <path_to_data> --exp_dir <path_to_output> --recon_model_checkpoint <path_to_reconstruction_model.pt> --model_type nongreedy --batch_size 4 --batches_step 4 --gamma 1 --lr_gamma 0.5 --scheduler_type multistep --accelerations 32 --acquisition_steps 28 --project <wandb_project_name> --wandb True
```
#### Brain
##### Base horizon greedy (1GPU)
```
CUDA_VISIBLE_DEVICES=0 python -m src.train_policy --dataset brain --data_path <path_to_data> --exp_dir <path_to_output> --resolution 256 --recon_model_checkpoint <path_to_reconstruction_model.pt> --num_chans 8 --sample_rate 0.2 --num_layers 5 --center_volume False --model_type greedy --project <wandb_project_name> --wandb True
```
##### Base horizon non-greedy (2GPU)
```
CUDA_VISIBLE_DEVICES=0,1 python -m src.train_policy --dataset brain --data_path <path_to_data> --exp_dir <path_to_output> --resolution 256 --recon_model_checkpoint <path_to_reconstruction_model.pt> --num_chans 8 --sample_rate 0.2 --num_layers 5 --center_volume False --model_type nongreedy --batch_size 4 --batches_step 4 --gamma 1 --lr_gamma 0.5 --scheduler_type multistep --project <wandb_project_name> --wandb True
```
##### Long horizon greedy (1GPU)
```
CUDA_VISIBLE_DEVICES=0 python -m src.train_policy --dataset brain --data_path <path_to_data> --exp_dir <path_to_output> --resolution 256 --recon_model_checkpoint <path_to_reconstruction_model.pt> --num_chans 8 --sample_rate 0.2 --num_layers 5 --center_volume False --model_type greedy --accelerations 32 --acquisition_steps 28 --project <wandb_project_name> --wandb True
```
##### Long horizon non-greedy (4GPU)
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.train_policy --dataset brain --data_path <path_to_data> --exp_dir <path_to_output> --resolution 256 --recon_model_checkpoint <path_to_reconstruction_model.pt> --num_chans 8 --sample_rate 0.2 --num_layers 5 --center_volume False --model_type nongreedy --batch_size 4 --batches_step 4 --gamma 1 --lr_gamma 0.5 --scheduler_type multistep --accelerations 32 --acquisition_steps 28 --project <wandb_project_name> --wandb True
```


## Evaluation
### Reconstruction models
All commands should be run from the repository root folder. To evaluate any reconstruction model on validation data, run:
```
CUDA_VISIBLE_DEVICES=0 python -m src.train_reconstruction --do_train False --data_path <path_to_data> --recon_model_checkpoint <path_to_reconstruction_model.pt> --partition val
```
### Policy models
To evaluate any policy model on test data, run:
```
CUDA_VISIBLE_DEVICES=0 python -m src.train_policy --do_train False --data_path <path_to_data> --recon_model_checkpoint <path_to_reconstruction_model.pt> --policy_model_checkpoint <path_to_policy_model.pt> --num_test_trajectories 8 --project <wandb_project_name> --wandb True 
```
### Baselines
The following commands are for running the baseline models reported in the paper (Random, NA Oracle, etc.). Presented are the commands for the Random baseline, and switching is as easy as setting `model_type` to a different value (see `run_baseline_models.py` for more detail).
Note that depending on your available GPU RAM, the default `batch_size` may need to be reduced to run the NA Oracle and Oracle baselines.

#### Knee
##### Base horizon random (1GPU)
```
CUDA_VISIBLE_DEVICES=0 python -m src.run_baseline_models --data_path <path_to_data> --recon_model_checkpoint <path_to_reconstruction_model.pt> --exp_dir <path_to_output> --project <wandb_project_name> --wandb True --model_type random
```
##### Long horizon random (1GPU)
```
CUDA_VISIBLE_DEVICES=0 python -m src.run_baseline_models --data_path <path_to_data> --recon_model_checkpoint <path_to_reconstruction_model.pt> --exp_dir <path_to_output> --project <wandb_project_name> --wandb True --accelerations 32 --acquisition_steps 28 --model_type random
```
#### Brain
##### Base horizon random (1GPU)
```
CUDA_VISIBLE_DEVICES=0 python -m src.run_baseline_models --dataset brain --resolution 256 --data_path <path_to_data> --recon_model_checkpoint <path_to_reconstruction_model.pt> --exp_dir <path_to_output> --project <wandb_project_name> --wandb True --sample_rate 0.2 --center_volume False --model_type random
```
##### Long horizon random (1GPU)
```
CUDA_VISIBLE_DEVICES=0 python -m src.run_baseline_models --dataset brain --resolution 256 --data_path <path_to_data> --recon_model_checkpoint <path_to_reconstruction_model.pt> --exp_dir <path_to_output> --project <wandb_project_name> --wandb True --sample_rate 0.2 --center_volume False --accelerations 32 --acquisition_steps 28 --model_type random
```



## SNR calculations
The `compute_snr.py` script allows for computation of SNR for multiple policy models at a time, given in the `<policy_model_dirs>` argument.

Notes: `<base_policy_model_dir>` in the below command will typically correspond to `exp_dir` in `train_policy.py`. `<policy_model_dirs>` are the datetime-stamped directory names where the policy models are stored by `train_policy.py`.

Be sure to specify the (Knee or Brain) data path and reconstruction model that corresponds to the policy models provided.

The command should be run from the repository root folder. 
```
CUDA_VISIBLE_DEVICES=0 python -m src.compute_snr --data_path <path_to_data> --recon_model_checkpoint <path_to_reconstruction_model.pt> --base_policy_model_dir <base_policy_model_dir> --policy_model_dir_list <policy_model_dirs>
```



## Reproducing figures
Figures can be reproduced using the IPython notebooks in the `notebooks` directory. Note that they require the specification of paths to data, reconstruction models, and policy models. With the exception of `visualise_policies.ipynb`, all notebooks require usage of the Weights and Biases API (and policy models stored using Weights and Biases).

- `visualise_policies.ipynb`: visualise average (marginal) policies and individual reconstructions.
- `evaluate_ssim.ipynb`: evaluate policy models on test data and compute averages.
- `mi_ent_curves.ipynb`: compute and visualise test data mutual information and entropies for policy models.
- `learning_curves.ipynb`: visualise learning curves on validation and train data.
- `split_data.ipynb`: utilities for data preparation (not for visualisation).


## Contributing

License TBD. 

[1]: https://www.wandb.com
