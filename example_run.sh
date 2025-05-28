#!/bin/bash

conda activate peaks_icml2025 

python main.py --experiment_name "cifar100_10k_PEAKS_seed_0" --output_dir "./experiments" --log_interval 1000 --seed 0 --method_name "PEAKS" --dataset "cifar100" --k 10000 --delta 8 --initial_training_size 1000 --initial_training_steps 100 --beta 128 --total_updates 2000 --learning_rate 0.001 --optimizer "sgd" --weight_decay 0.0001  --selection_p 20.0 --p 100 --class_coeff 1 