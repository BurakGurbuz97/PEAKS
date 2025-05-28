import torch
from Source.utils import setup_logging, save_experiment_config, create_experiment_dir
import random
import numpy as np
from args import parse_args
from incremental import main_incremental


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make cudnn deterministic (slightly lower performance but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)

    # Setup experiment
    exp_dir = create_experiment_dir(args.output_dir, args.experiment_name)
    save_experiment_config(args, exp_dir)
    logger = setup_logging(exp_dir)
    logger.info(f"Starting experiment: {args.experiment_name}")

    main_incremental(args, exp_dir, logger)
