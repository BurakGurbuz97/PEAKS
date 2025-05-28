from typing import Dict, Tuple
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from Source.model_handler import ViTHandler
from Source.data_handler import DataPoolManager
from Source.utils import MetricsLogger, pick_optimizer, pick_method
import torch
import torch.nn.functional as F
import pickle
import os



class Trainer:
    def __init__(
        self,
        model_handler: ViTHandler,
        data_manager: DataPoolManager,
        test_loader: DataLoader,
        optimizer_params: Dict[str, float],
        device: str,
        exp_dir: str,
        log_interval: int,
        logger: logging.Logger,
        args
    ) -> None:
        """Initialize trainer."""
        self.model_handler = model_handler
        self.data_manager = data_manager
        self.test_loader = test_loader
        self.device = device
        self.exp_dir = exp_dir
        self.log_interval = log_interval
        self.logger = logger
        self.args = args

        self.optimizer = pick_optimizer(args.optimizer, model_handler.model.parameters(), optimizer_params)
        self.criterion = nn.CrossEntropyLoss()
        self.metrics_logger = MetricsLogger(exp_dir)

        # Initialize parameters
        self.u = 0  # Update clock
        self.s = 0  # Sleep clock
        self.U_remaining = args.total_updates
        self.delta = args.delta
        self.beta = args.beta
        self.p = args.p
        self.method = pick_method(args.method_name, model_handler, data_manager, args, self.logger)

    def train_initial(self, initial_train_loader: DataLoader, num_steps: int) -> None:
        """Train on initial training set T_0 for specified number of steps."""
        self.model_handler.model.train()
        step = 0
        while step < num_steps and self.U_remaining > 0:
            for batch in initial_train_loader:
                if step >= num_steps or self.U_remaining <= 0:
                    break
                self.model_handler.train_batch(batch, self.criterion, self.optimizer)
                self.U_remaining -= 1
                self.u += 1
                step += 1
                # Update logging within training loops (e.g., in train_initial)
                if self.u % self.log_interval == 0:
                    self.progress_log()

        self.logger.info(f"Initial training completed. Updates remaining: {self.U_remaining}")
        if self.method:
            self.method.sleep()
            self.logger.info("Initial training sleep step completed.")


    def train_incremental(self) -> None:
        """Train using incremental data selection framework."""
        while self.U_remaining > 0:
            # Example Selection
            if self.data_manager.termination_condition_met(self.delta):
                self.logger.info("Termination condition met.")
                break

            selected_indices = self.method.select_examples()
            update_indices = self.method.form_batch(selected_indices)
            batch_data = self.data_manager.get_data(update_indices)
            inputs = torch.stack([item[0] for item in batch_data])
            labels = torch.tensor([item[1] for item in batch_data])
  
            # Train on the batch
            self.model_handler.train_batch((inputs, labels), self.criterion, self.optimizer)
            
            self.U_remaining -= 1
            self.u += 1
            if self.U_remaining <= 0:
                self.logger.info("Update budget exhausted.")
                break

            # Update logging within training loops (e.g., in train_initial)
            if self.u % self.log_interval == 0:
                self.progress_log()

            # Sleep Condition
            if self.u % self.p == 0:
                self.method.sleep()
                self.s += 1
                self.logger.info(f"Sleep step completed. Sleep clock s = {self.s}")

        
        counts = self.method.sampling_counts
        #pickle the counts using args.output_dir, args.experiment_name
        path = os.path.join(self.args.output_dir, self.args.experiment_name, "sampling_counts.pkl")
        with open(path, "wb") as f:
            pickle.dump(counts, f)

        if self.args.store_selection_info:
            path = os.path.join(self.args.output_dir, self.args.experiment_name, "selection_info.pkl")
            with open(path, "wb") as f:
                pickle.dump(self.method.selection_info, f)

    def fine_tune(self, final_train_loader: DataLoader) -> None:
        """Fine-tune the model on T_end if updates remain."""
        self.model_handler.model.train()
        while self.U_remaining > 0:
            for batch in final_train_loader:
                if self.U_remaining <= 0:
                    self.logger.info("Update budget exhausted during fine-tuning.")
                    break
                self.model_handler.train_batch(batch, self.criterion, self.optimizer)
                self.U_remaining -= 1
                self.u += 1

                # Update logging within training loops (e.g., in train_initial)
                if self.U_remaining > 0 and self.u % self.log_interval == 0:
                    self.progress_log()
             
        self.progress_log()

    def evaluate(self) -> Tuple[float, float, float, float, float, float]:
        """Evaluate model on validation and test sets."""
        train_loader = DataLoader(
            self.data_manager.get_dataset_T_end(),
            batch_size=self.args.beta,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        self.model_handler.model.eval()
        with torch.no_grad():
            train_losses, train_corrects = [], []
            for inputs, labels in train_loader:
                inputs = inputs.to(self.model_handler.device)
                labels = labels.to(self.model_handler.device)
                outputs = self.model_handler.model(inputs)
                loss = F.cross_entropy(outputs, labels, reduction='none')
                train_losses.extend(loss.cpu().numpy())
                preds = outputs.argmax(dim=1, keepdim=True)
                train_corrects.extend(preds.eq(labels.view_as(preds)).cpu().numpy())
            avg_train_loss = sum(train_losses) / len(train_losses)
            avg_train_acc = sum(train_corrects) / len(train_corrects)

            # Evaluate on test set
            test_losses, test_corrects = [], []
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.model_handler.device)
                labels = labels.to(self.model_handler.device)
                outputs = self.model_handler.model(inputs)
                loss = F.cross_entropy(outputs, labels, reduction='none')
                test_losses.extend(loss.cpu().numpy())
                preds = outputs.argmax(dim=1, keepdim=True)
                test_corrects.extend(preds.eq(labels.view_as(preds)).cpu().numpy())
            avg_test_loss = sum(test_losses) / len(test_losses)
            avg_test_acc = sum(test_corrects) / len(test_corrects)

        return avg_train_loss, float(avg_train_acc), avg_test_loss, float(avg_test_acc)


    def progress_log(self):
        train_loss, train_acc, test_loss, test_acc = self.evaluate()
        examples_selected = self.data_manager.get_total_selected_examples()
        self.metrics_logger.log_metrics(
            self.u, train_loss, train_acc,  test_loss, test_acc,
            examples_selected=examples_selected,
            is_final=False
        )
        self.logger.info(
            f"Update {self.u} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, "
            f"Examples Selected: {examples_selected}"
        )


    def updates_remaining(self) -> int:
        """Return the number of updates remaining."""
        return self.U_remaining
