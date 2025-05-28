import argparse
import torch
from multiprocessing import cpu_count

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='ViT Incremental Data Selection Framework')

    # Experiment Arguments
    parser.add_argument('--experiment_name', type=str, default="test")
    parser.add_argument('--output_dir', type=str, default='./experiments')
    parser.add_argument('--log_interval', type=int, default=400)
    parser.add_argument('--seed', type=int, default=1)

    # Dataset Arguments
    parser.add_argument('--dataset', type=str, choices=['cifar100', 'food101', 'food101-noise', 'webvision'], default='food101-noise')

    # Incremental Data Selection Arguments
    parser.add_argument('--k', type=int, default=2500,help='Total number of training samples')
    parser.add_argument('--delta', type=int, default=2, help='Number of new examples to select in each example selection step')

    parser.add_argument('--initial_training_size', type=int, default=1000) # Used for the step-1 (initialization) of IDS
    parser.add_argument('--initial_training_steps', type=int, default=100) # Used for the step-1 (initialization) of IDS
    parser.add_argument('--beta', type=int, default=128, help='Total batch size for model update')
    parser.add_argument('--total_updates', type=int, default=2_000, help='Total number of model updates allowed') # Initialization + Data Selection + Final Training

    # Optimization Arguments
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam', 'adamw'])
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    
    # Method argument
    parser.add_argument('--method_name', type=str, default='grad_norm', choices=['random',
                                                                                'random_balanced',
                                                                                "embedding",
                                                                                "PEAKS",
                                                                                "error",
                                                                                "uncertainty",
                                                                                "grad_norm",
                                                                                "max_post_prob"]) # this is wrong and low confidence selection (maximum posterior probability)
    
    # General arguments
    parser.add_argument('--selection_p', type=float, default=20.0, help='Percentage of data to select using percentile trick')
    parser.add_argument('--p', type=int, default=100, help='Number of updates after which cache refresh occurs') # this is \tau from the paper
   
    # Our arguments
    parser.add_argument('--class_coeff', type=int, default=1) # boolean (0 or 1) to use class balancing trick at Section 4.1

    # Baseline Arguments
    parser.add_argument('--embedding_type', type=str, default="hard", choices=['hard', 'easy'])
    
    # use for moderate selection e.g. embedding_type = "hard" and embedding_noise_adjust = 40 and --selection_p = 20
    # moderate selection from 40%-60% of the embedding distance
    parser.add_argument('--embedding_noise_adjust', type=float, default=40.0)
    parser.add_argument('--method_sampling', type=str, default="random", choices=['random', 'count'],
                        help='Sampling method for selecting examples from the pool. "random" for random sampling, "count" for counting based on selection scores.')

    # Architecture Details
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224_dino') 
    parser.add_argument('--base_vit_config', type=str, default='VIT_16_BASE_CONFIG') 
    parser.add_argument('--pretrained', type=int, default=1)
    parser.add_argument('--freeze_config_path', type=str, default='',
                        help='Path to the parameter freeze configuration file.')
    parser.add_argument('--store_selection_info', type=int, default=0)

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu' # automatically set device based on availability
    args.num_workers = min(cpu_count(), 16) # Use up to 16 workers or the number of CPU cores, whichever is smaller
    return args
