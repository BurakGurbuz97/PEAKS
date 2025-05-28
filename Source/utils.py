import os
import json
import logging
import pandas as pd
import shutil
from typing import Dict, Any, Optional, List
from Source.acquisition import RandomMethod, EmbeddingMethod, PEAKS, RandomBalancedMethod, MaxPostMethod,  ErrorMethod, UncertaintyMethod, GradNormMethod
import torch


def pick_method(method_name, model_handler, data_manager, args, logger):
    if method_name == 'random':
        method = RandomMethod(model_handler, data_manager, args)
    elif method_name == 'random_balanced':
        method = RandomBalancedMethod(model_handler, data_manager, args)
    elif method_name == 'PEAKS':
        method = PEAKS(model_handler, data_manager, args, logger, len(data_manager.full_dataset.classes))
    elif method_name == 'embedding':
        method = EmbeddingMethod(model_handler, data_manager, args, logger, len(data_manager.full_dataset.classes))
    elif method_name == 'error':
        method = ErrorMethod(model_handler, data_manager, args, logger, len(data_manager.full_dataset.classes))
    elif method_name == 'uncertainty':
        method = UncertaintyMethod(model_handler, data_manager, args, logger, len(data_manager.full_dataset.classes))
    elif method_name == 'grad_norm':
        method = GradNormMethod(model_handler, data_manager, args, logger, len(data_manager.full_dataset.classes))
    elif method_name == 'max_post_prob':
        method = MaxPostMethod(model_handler, data_manager, args, logger, len(data_manager.full_dataset.classes))
    else:
        raise NotImplementedError(f"Method '{method_name}' is not implemented.")
    
    return method

def pick_optimizer(optimizer_name, model_parameters, optimizer_params):
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            model_parameters,
            lr=optimizer_params['lr'],
            weight_decay=optimizer_params['weight_decay'],
            momentum=optimizer_params['momentum']
            )
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            model_parameters,
            lr=optimizer_params['lr'],
            weight_decay=optimizer_params['weight_decay']
        )
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            model_parameters,
            lr=optimizer_params['lr'],
            weight_decay=optimizer_params['weight_decay']
        )
    else:
        raise NotImplementedError(f"Optimizer '{optimizer_name}' is not implemented.")
    
    return optimizer

class MetricsLogger:
    def __init__(self, exp_dir: str) -> None:
        self.exp_dir = exp_dir
        self.metrics: List[Dict[str, Any]] = []
        self.csv_path = os.path.join(exp_dir, 'metrics.csv')
        
        # Load existing metrics if resuming
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)
            self.metrics = df.to_dict('records')
        else:
            self.metrics = []

    def log_metrics(
        self,
        step: int,
        train_loss: float,
        train_acc: float,
        val_acc: Optional[float] = None,
        test_loss: Optional[float] = None,
        test_acc: Optional[float] = None,
        examples_selected: int = 0,
        is_final: bool = False
    ) -> None:
        """Log and save training/testing metrics.

        Args:
            step: The current training step.
            train_loss: Training loss.
            train_acc: Training accuracy.
            test_loss: Testing loss.
            test_acc: Testing accuracy.
            examples_selected: Number of examples selected.
            is_final: Whether these are the final metrics.
        """
        metrics = {
            'step': step,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'examples_selected': examples_selected,
            'is_final': is_final
        }
        self.metrics.append(metrics)
        df = pd.DataFrame(self.metrics)
        df.to_csv(self.csv_path, index=False)

    def get_metrics(self) -> pd.DataFrame:
        """Retrieve the logged metrics as a DataFrame.

        Returns:
            Pandas DataFrame containing the metrics.
        """
        return pd.DataFrame(self.metrics)
    
    def setup_logging(exp_dir: str) -> logging.Logger:
        """Set up logging for the experiment.

        Args:
            exp_dir: Path to the experiment directory.

        Returns:
            Configured logger.
        """
        logger = logging.getLogger('experiment')
        logger.setLevel(logging.INFO)

        # Clear existing handlers
        if logger.hasHandlers():
            logger.handlers.clear()

        # Ensure the FileHandler is in append mode
        log_file = os.path.join(exp_dir, 'experiment.log')
        fh = logging.FileHandler(log_file, mode='a')  # 'a' for append
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(message)s')

        for handler in [fh, ch]:
            handler.setLevel(logging.INFO)
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger


def create_experiment_dir(base_dir: str, experiment_name: str, overwrite=False) -> str:
    """Create a directory for the experiment.

    Args:
        base_dir: Base directory path.
        experiment_name: Name of the experiment.
        overwrite: Whether to overwrite the existing directory.

    Returns:
        The path to the experiment directory.
    """
    exp_dir = os.path.join(base_dir, experiment_name)

    if os.path.exists(exp_dir):
        if overwrite:
            shutil.rmtree(exp_dir)
            os.makedirs(exp_dir, exist_ok=True)
        else:
            # Do not delete existing directory when resuming
            pass
    else:
        os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def save_experiment_config(args: Any, exp_dir: str) -> None:
    """Save the experiment configuration to a JSON file.

    Args:
        args: Arguments or configuration object.
        exp_dir: Path to the experiment directory.
    """
    config_path = os.path.join(exp_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)


def setup_logging(exp_dir: str) -> logging.Logger:
    """Set up logging for the experiment.

    Args:
        exp_dir: Path to the experiment directory.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger('experiment')
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(os.path.join(exp_dir, 'experiment.log'))
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    for handler in [fh, ch]:
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger