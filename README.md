# Experimental design for MRI by greedy policy search
This repository is the official implementation of: [Experimental design for MRI by greedy policy search]() (NeurIPS, 2020).

> Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

> Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

Data is fastMRI, explain preprocessing! And hardcoded folder names.

Data should be stored as follows, as the top-level folders are hardcoded into our data loaders.
```
<path_to_data>
  singlecoil_train/
  singlecoil_val/
  singlecoil_test/
```

This corresponds somewhat to the default download settings of the fastMRI data repository. For both Knee and Brain datasets the original test data contains not ground truths, and so we construct a new `singlecoil_test` from `singlecoil_train`, as explained in the paper.
The default Brain data is `multicoil_` instead of `singlecoil_`. Note that we do use not use the multicoil k-space and instead construct singlecoil k-space from the ground truth images. To save on I/O, we recommend removing the multicoil k-space from the `.h5` files. For naming consistency, we have also renamed `multicoil_` to `singlecoil_` for Brain data.


## Training
All command should be run from the repository root folder. Logging is done using Tensorboard.
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
#### Knee
##### Base horizon greedy (1GPU)
```
CUDA_VISIBLE_DEVICES=0 python -m src.train_policy --data_path <path_to_data> --exp_dir <path_to_output> --recon_model_checkpoint <path_to_reconstruction_model.pt> --model_type greedy --accelerations 8 --acquisition_steps 16 --project <wandb_project_name> --wandb True
```
##### Base horizon non-greedy (1GPU)
```
CUDA_VISIBLE_DEVICES=0 python -m src.train_policy --data_path <path_to_data> --exp_dir <path_to_output> --recon_model_checkpoint <path_to_reconstruction_model.pt> --model_type nongreedy --batch_size 4 --batches_step 4 --gamma 1 --lr_gamma 0.5 --scheduler_type multistep --accelerations 8 --acquisition_steps 16 --project <wandb_project_name> --wandb True
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
CUDA_VISIBLE_DEVICES=0 python -m src.train_policy --dataset brain --data_path <path_to_data> --exp_dir <path_to_output> --resolution 256 --recon_model_checkpoint <path_to_reconstruction_model.pt> --num_chans 8 --sample_rate 0.2 --num_layers 5 --center_volume False --model_type greedy --accelerations 8 --acquisition_steps 16 --project <wandb_project_name> --wandb True
```
##### Base horizon non-greedy (2GPU)
```
CUDA_VISIBLE_DEVICES=0,1 python -m src.train_policy --dataset brain --data_path <path_to_data> --exp_dir <path_to_output> --resolution 256 --recon_model_checkpoint <path_to_reconstruction_model.pt> --num_chans 8 --sample_rate 0.2 --num_layers 5 --center_volume False --model_type nongreedy --batch_size 4 --batches_step 4 --gamma 1 --lr_gamma 0.5 --scheduler_type multistep --accelerations 8 --acquisition_steps 16 --project <wandb_project_name> --wandb True
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
To evaluate any reconstruction model on validation data, run:
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
### Brain
##### Base horizon random (1GPU)
```
CUDA_VISIBLE_DEVICES=0 python -m src.run_baseline_models --dataset brain --resolution 256 --data_path <path_to_data> --recon_model_checkpoint <path_to_reconstruction_model.pt> --exp_dir <path_to_output> --project <wandb_project_name> --wandb True --sample_rate 0.2 --center_volume False --model_type random
```
##### Long horizon random (1GPU)
```
CUDA_VISIBLE_DEVICES=0 python -m src.run_baseline_models --dataset brain --resolution 256 --data_path <path_to_data> --recon_model_checkpoint <path_to_reconstruction_model.pt> --exp_dir <path_to_output> --project <wandb_project_name> --wandb True --sample_rate 0.2 --center_volume False --accelerations 32 --acquisition_steps 28 --model_type random
```

> Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## SNR calculations

## Reproducing figures

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

> Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

> Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

> Pick a licence and describe how to contribute to your code repository. 

[1]: https://www.wandb.com