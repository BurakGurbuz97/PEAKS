import random
from typing import List
from Source.data_handler import DataPoolManager
from Source.model_handler import ViTHandler
import torch
import numpy as np
import time
from torch.utils.data import DataLoader, IterableDataset
from scipy.stats import percentileofscore
import torch.nn.functional as F
import math


class InfiniteRandomSampler(IterableDataset):
    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.dataset_size = len(dataset)
        # We want this to be fair across all methods so we generate a new random object with a fixed seed
        self.random = random.Random(seed + 1000000) # Add a large number to seed to avoid overlap with other seeds
    
    def __iter__(self):
        while True:
            idx = self.random.randint(0, self.dataset_size - 1)
            yield idx, self.dataset[idx][0], self.dataset[idx][1]

class Method:
    """Generic method class for acquisition, batch forming, and sleep operations."""

    def __init__(self, model_handler: ViTHandler, data_manager: DataPoolManager, args) -> None:
        self.model_handler = model_handler
        self.data_manager = data_manager
        self.args = args
        self.data_loader = iter(DataLoader(InfiniteRandomSampler(self.data_manager.full_dataset, args.seed),
                                           batch_size=64, num_workers=1))
        self.sampling_counts = np.zeros(len(self.data_manager.full_dataset)) - 1
        self.sampling_counts[list(self.data_manager.T_s_indices)] = (self.args.initial_training_steps * self.args.beta) // self.args.initial_training_size
        self.error_probs = None
        self.selection_info = []

    def select_examples(self) -> List[int]:
        """Example selection step."""
        raise NotImplementedError

    def form_batch(self, selected_indices: List[int]) -> List[int]:
        if self.args.method_sampling == 'random':
            return self.form_batch_random(selected_indices)
        elif self.args.method_sampling == 'count':
            return self.form_batch_count(selected_indices)
        else:
            raise NotImplementedError(f"Sampling method '{self.args.method_sampling}' is not implemented.")

    def form_batch_random(self, selected_indices: List[int]) -> List[int]:
        """Form batch by sampling from T_s uniformly at random."""
        num_T_s_samples = self.args.beta - len(selected_indices)
        T_s_indices = self.data_manager.sample_from_T_s(num_T_s_samples)
        for idx in T_s_indices:
            self.sampling_counts[idx] += 1
        update_indices = T_s_indices + selected_indices
        return update_indices
    
    def form_batch_count(self, selected_indices: List[int]) -> List[int]:
        num_T_s_samples = self.args.beta - len(selected_indices)
        T_s_indices = self.count_sample(num_T_s_samples, selected_indices)
        for idx in T_s_indices:
            self.sampling_counts[idx] += 1
        update_indices = list(T_s_indices) + list(selected_indices)
        return update_indices
    
    def count_sample(self, num_samples: int, already_selected: int) -> List[int]:
        mask = (self.sampling_counts > 0)
        
        # Already selected should have all False in mask check this if not raise error
        if np.any(mask[already_selected]):
            raise ValueError("Already selected indices are in mask -- something wrong with the logic")

        available_indices = np.where(mask)[0]

        if len(available_indices) < num_samples:
            raise ValueError(f"Available indices ({len(available_indices)}) less than required samples ({num_samples})")
        
        available_counts = self.sampling_counts[available_indices]
        inv_counts = 1.0 / (available_counts)
        probabilities = inv_counts / np.sum(inv_counts)
        selected_indices = np.random.choice(available_indices, size=num_samples, p=probabilities, replace=False)
        return selected_indices

    def update_S_counts(self) -> None:
        s_indices = self.data_manager.S_indices
        for idx in s_indices:
            self.sampling_counts[idx] = 1
        
    def sleep(self) -> None:
        """Move examples from S to T_s."""
        self.update_S_counts()
        self.data_manager.move_S_to_T_s()
            

class RandomMethod(Method):
    """Random method for acquisition, batch forming, and sleep operations."""

    def __init__(self, model_handler: ViTHandler, data_manager: DataPoolManager, args) -> None:
        super().__init__(model_handler, data_manager, args)

    def select_examples(self) -> List[int]:
        """Randomly select examples with a fixed probability."""
        selected_indices = []
        while len(selected_indices) < self.args.delta:
            batch_data = next(self.data_loader)
            batch_indices = [int(item) for item in batch_data[0]]
            for i in batch_indices:
                if (i in self.data_manager.pool_indices) and (i not in selected_indices):
                    selected_indices.append(i)
                if len(selected_indices) >= self.args.delta:
                    break
        self.data_manager.add_to_S(selected_indices)
        return selected_indices

class RandomBalancedMethod(Method):

    def __init__(self, model_handler: ViTHandler, data_manager: DataPoolManager, args) -> None:
        super().__init__(model_handler, data_manager, args)
        self.count_per_class = math.ceil(self.args.k  / len(self.data_manager.class_counts))


    def select_examples(self) -> List[int]:
        selected_indices = []
        while len(selected_indices) < self.args.delta:
            batch_data = next(self.data_loader)
            batch_indices = [int(item) for item in batch_data[0]]
            labels = torch.tensor([item for item in batch_data[2]])

            for list_i, i in enumerate(batch_indices):
                if (i in self.data_manager.pool_indices) and (i not in selected_indices) and (self.data_manager.class_counts[int(labels[list_i])] <= self.count_per_class):
                    self.data_manager.add_to_S([i])
                if len(selected_indices) >= self.args.delta:
                    break

        
        return selected_indices
    
class GradNormMethod(Method):
    def __init__(self, model_handler: ViTHandler, data_manager: DataPoolManager, args, logger, num_classes) -> None:
        super().__init__(model_handler, data_manager, args)
        self.time_spend = 0
        self.num_classes = num_classes
        self.logger = logger
        self.selection_round = 0
        self.selection_info = []
        self.decisions = []
        self.recent_norms = []
        self.criterion = torch.nn.CrossEntropyLoss()

    def select_examples(self) -> List[int]:
        current_time = time.time()
        selected_indices = []

        while len(selected_indices) < self.args.delta:
            if self.data_manager.get_pool_size() == 0:
                break

            batch_data = next(self.data_loader)
            batch_indices = [int(item) for item in batch_data[0]]
            inputs = torch.stack([item for item in batch_data[1]]).to(self.model_handler.device)  # Shape: [batch_size, C, H, W]
            labels = torch.tensor([item for item in batch_data[2]]).to(self.model_handler.device)  # Shape: [batch_size]

            for i, batch_idx in enumerate(batch_indices):
                if (batch_idx in self.data_manager.pool_indices) and (batch_idx not in selected_indices):
                    select, importance = self.check_rule(inputs[i], labels[i])
                    if select:
                        selected_indices.append(batch_idx)
                    self.decisions.append(select)
                    self.selection_info.append((self.selection_round, batch_idx, self.decisions[-1],
                                                0 , labels[i].item(), importance))
                
                if len(selected_indices) >= self.args.delta:
                    break
        
        self.data_manager.add_to_S(selected_indices)
        self.time_spend += time.time() - current_time
        return selected_indices
    
    def check_rule(self, x, y):
        grad_norm = self.compute_grad_norm(x, y)
        self.recent_norms.append(grad_norm)
        select = 100.0 - self.args.selection_p <= percentileofscore(self.recent_norms, grad_norm)
        return select, grad_norm
    
    def compute_grad_norm(self, x, y):
        self.model_handler.model.eval()
        output = self.model_handler.model(x.unsqueeze(0))
        loss = self.criterion(output, y.unsqueeze(0))

        grads = torch.autograd.grad(loss, self.model_handler.model.parameters(),
                                    create_graph=False, retain_graph=False)
    
        grad_norm = 0.0
        for g in grads:
            grad_norm += g.norm(2).item() ** 2
        
        grad_norm = grad_norm ** 0.5
        
        return grad_norm
    
    def sleep(self) -> None:
        self.update_S_counts()
        self.selection_round = self.selection_round + 1
        self.data_manager.move_S_to_T_s()
        self.logger.info(f"Time spend: {self.time_spend:.4f}")
        self.time_spend = 0

        self.logger.info(f"Decision rate: {np.mean(self.decisions):.4f}")
        self.decisions = []
        self.recent_norms = []


    
class ErrorMethod(Method):

    def __init__(self, model_handler: ViTHandler, data_manager: DataPoolManager, args, logger, num_classes) -> None:
        super().__init__(model_handler, data_manager, args)
        self.time_spend = 0
        self.num_classes = num_classes
        self.logger = logger
        self.selection_round = 0
        self.selection_info = []
        self.decisions = []
        self.recent_errors = []

    def select_examples(self) -> List[int]:
        current_time = time.time()
        selected_indices = []

        while len(selected_indices) < self.args.delta:
            if self.data_manager.get_pool_size() == 0:
                break

            batch_data = next(self.data_loader)
            batch_indices = [int(item) for item in batch_data[0]]
            inputs = torch.stack([item for item in batch_data[1]]).to(self.model_handler.device)  # Shape: [batch_size, C, H, W]
            labels = torch.tensor([item for item in batch_data[2]]).to(self.model_handler.device)  # Shape: [batch_size]


            self.model_handler.model.eval()
            with torch.no_grad():
                penultimate = self.model_handler.model.get_penultimate(inputs).detach()
                logits = self.model_handler.model.head(penultimate)
                softmax = F.softmax(logits, dim=1)
                y_train_one_hot = F.one_hot(labels, self.num_classes).float().to(self.model_handler.device)

            for i, batch_idx in enumerate(batch_indices):
                if (batch_idx in self.data_manager.pool_indices) and (batch_idx not in selected_indices):
                    select, importance = self.check_rule(y_train_one_hot[i], softmax[i])
                    if select:
                        selected_indices.append(batch_idx)
                    self.decisions.append(select)
                    self.selection_info.append((self.selection_round, batch_idx, self.decisions[-1],
                                                softmax[i].cpu().numpy() , labels[i].item(), importance))
                    
                
                if len(selected_indices) >= self.args.delta:
                    break
        

        self.data_manager.add_to_S(selected_indices)
        self.time_spend += time.time() - current_time
        return selected_indices

    def check_rule(self, y_train_one_hot, softmax):
        # L2 norm of the error
        error_norm = float((y_train_one_hot - softmax).norm(p=2))
        self.recent_errors.append(error_norm)
        select = 100.0 - self.args.selection_p <= percentileofscore(self.recent_errors, error_norm)
        return select, error_norm
    
    def sleep(self) -> None:
        self.update_S_counts()
        self.selection_round = self.selection_round + 1
        self.data_manager.move_S_to_T_s()
        self.logger.info(f"Time spend: {self.time_spend:.4f}")
        self.time_spend = 0

        self.logger.info(f"Decision rate: {np.mean(self.decisions):.4f}")
        self.decisions = []
        self.recent_errors = []
        

class UncertaintyMethod(Method):

    def __init__(self, model_handler: ViTHandler, data_manager: DataPoolManager, args, logger, num_classes) -> None:
        super().__init__(model_handler, data_manager, args)
        self.time_spend = 0
        self.num_classes = num_classes
        self.logger = logger
        self.selection_round = 0
        self.selection_info = []
        self.decisions = []
        self.recent_uncert = []

    def select_examples(self) -> List[int]:
        current_time = time.time()
        selected_indices = []

        while len(selected_indices) < self.args.delta:
            if self.data_manager.get_pool_size() == 0:
                break

            batch_data = next(self.data_loader)
            batch_indices = [int(item) for item in batch_data[0]]
            inputs = torch.stack([item for item in batch_data[1]]).to(self.model_handler.device)  # Shape: [batch_size, C, H, W]
            labels = torch.tensor([item for item in batch_data[2]]).to(self.model_handler.device)  # Shape: [batch_size]


            self.model_handler.model.eval()
            with torch.no_grad():
                penultimate = self.model_handler.model.get_penultimate(inputs).detach()
                logits = self.model_handler.model.head(penultimate)
                softmax = F.softmax(logits, dim=1)

            for i, batch_idx in enumerate(batch_indices):
                if (batch_idx in self.data_manager.pool_indices) and (batch_idx not in selected_indices):
                    select, importance = self.check_rule(softmax[i])
                    if select:
                        selected_indices.append(batch_idx)
                    self.decisions.append(select)
                    self.selection_info.append((self.selection_round, batch_idx, self.decisions[-1],
                                                softmax[i].cpu().numpy() , labels[i].item(), importance))
                    
                
                if len(selected_indices) >= self.args.delta:
                    break
        

        self.data_manager.add_to_S(selected_indices)
        self.time_spend += time.time() - current_time
        return selected_indices

    def check_rule(self, softmax):
        least_confidence = 1.0 - softmax.max().item()
        self.recent_uncert.append(least_confidence)
        select = 100.0 - self.args.selection_p <= percentileofscore(self.recent_uncert, least_confidence)
        return select, least_confidence
    
    def sleep(self) -> None:
        self.update_S_counts()
        self.selection_round = self.selection_round + 1
        self.data_manager.move_S_to_T_s()
        self.logger.info(f"Time spend: {self.time_spend:.4f}")
        self.time_spend = 0

        self.logger.info(f"Decision rate: {np.mean(self.decisions):.4f}")
        self.decisions = []
        self.recent_uncert = []

class EmbeddingMethod(Method):

    def __init__(self, model_handler: ViTHandler, data_manager: DataPoolManager, args, logger, num_classes) -> None:
        super().__init__(model_handler, data_manager, args)
        self.time_spend = 0
        self.num_classes = num_classes
        self.logger = logger
        self.selection_round = 0
        self.selection_info = []
        self.decisions = []
        self.recent_distances = []


    def select_examples(self) -> List[int]:
        current_time = time.time()
        selected_indices = []
        class_neuron_weights = self.model_handler.model.head.weight.data.cpu().numpy()

        while len(selected_indices) < self.args.delta:
            if self.data_manager.get_pool_size() == 0:
                break

            batch_data = next(self.data_loader)
            batch_indices = [int(item) for item in batch_data[0]]
            inputs = torch.stack([item for item in batch_data[1]]).to(self.model_handler.device)  # Shape: [batch_size, C, H, W]
            labels = torch.tensor([item for item in batch_data[2]]).to(self.model_handler.device)  # Shape: [batch_size]

            self.model_handler.model.eval()
            with torch.no_grad():
                penultimate = self.model_handler.model.get_penultimate(inputs)
                softmax = F.softmax(self.model_handler.model.head(penultimate), dim=1)

            for i, batch_idx in enumerate(batch_indices):
                if (batch_idx in self.data_manager.pool_indices) and (batch_idx not in selected_indices):
                    cosine_distance = self._cosine_distance(penultimate[i].cpu().numpy(), class_neuron_weights[int(labels[i]), :])
                    self.recent_distances.append(cosine_distance)
                    select = self._check_rule(cosine_distance)
                    if select:
                        selected_indices.append(batch_idx)
                    self.decisions.append(select)
                    self.selection_info.append((self.selection_round, batch_idx, self.decisions[-1],
                                                softmax[i].cpu().numpy() , labels[i].item(), cosine_distance))

                if len(selected_indices) >= self.args.delta:
                    break                

        self.data_manager.add_to_S(selected_indices)
        self.time_spend += time.time() - current_time
        return selected_indices


    def _check_rule(self, distance) -> bool:
        percentile_score = percentileofscore(self.recent_distances, distance)
        if self.args.embedding_type == "hard":
            return (100.0 - self.args.embedding_noise_adjust >= percentile_score >= 100.0 - self.args.embedding_noise_adjust - self.args.selection_p)
        elif self.args.embedding_type == "easy":
            return (self.args.selection_p >= percentile_score) 
        else:
            raise ValueError(f"Invalid embedding type {self.args.embedding_type}")


    def sleep(self) -> None:
        self.update_S_counts()
        self.selection_round = self.selection_round + 1
        self.data_manager.move_S_to_T_s()
        self.logger.info(f"Time spend: {self.time_spend:.4f}")
        self.time_spend = 0
        self.logger.info(f"Decision rate: {np.mean(self.decisions):.4f}")
        self.decisions = []
        self.recent_distances = []


    def _cosine_distance(self, a, b):
        return 1.0 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


    

class PEAKS(Method):

    def __init__(self, model_handler: ViTHandler, data_manager: DataPoolManager, args, logger, num_classes) -> None:
        super().__init__(model_handler, data_manager, args)
        self.logger = logger
        self.num_classes = num_classes
        self.decisions = []
        self.selection_round = 0
        self.selection_info = []
        self.time_spend = 0
        self.recent_importance = []

    def select_examples(self) -> List[int]:
        current_time = time.time()
        selected_indices = []

        while len(selected_indices) < self.args.delta:
            if self.data_manager.get_pool_size() == 0:
                break

            batch_data = next(self.data_loader)
            batch_indices = [int(item) for item in batch_data[0]]
            inputs = torch.stack([item for item in batch_data[1]]).to(self.model_handler.device)  # Shape: [batch_size, C, H, W]
            labels = torch.tensor([item for item in batch_data[2]]).to(self.model_handler.device)  # Shape: [batch_size]

            # Put model in evaluation mode
            self.model_handler.model.eval()
            with torch.no_grad():
                penultimate = self.model_handler.model.get_penultimate(inputs).detach()
                logits = self.model_handler.model.head(penultimate)
                softmax = F.softmax(logits, dim=1)
                y_train_one_hot = F.one_hot(labels, self.num_classes).float().to(self.model_handler.device)

            for i, batch_idx in enumerate(batch_indices):
                if (batch_idx in self.data_manager.pool_indices) and (batch_idx not in selected_indices):
                    select, importance = self.check_rule(labels[i].item(), y_train_one_hot[i], softmax[i], logits[i])
                    if select:
                        selected_indices.append(batch_idx)
                    self.decisions.append(select)
                    self.selection_info.append((self.selection_round, batch_idx, self.decisions[-1],
                                                softmax[i].cpu().numpy() , labels[i].item(), importance))
                    
                
                if len(selected_indices) >= self.args.delta:
                    break
        
        self.data_manager.add_to_S(selected_indices)
        self.time_spend += time.time() - current_time
        return selected_indices
    
    def check_rule(self, y_train, y_train_one_hot, softmax, logits):
        error = (y_train_one_hot - softmax).abs().sum()
        if self.args.class_coeff:
            if self.data_manager.class_counts[y_train] == 0:
                return True, 99999999.0 # Select if class is empty
            y_train_weight = 1.0 / self.data_manager.class_counts[y_train]
        else:
            y_train_weight = 1.0

        kernel = float(logits[y_train])
        importance = (y_train_weight * kernel * error).detach().cpu().item()
        self.recent_importance.append(importance)
        select = 100.0 - self.args.selection_p <= percentileofscore(self.recent_importance, importance)
        return select, importance

    
    def sleep(self) -> None:
        self.update_S_counts()
        self.selection_round = self.selection_round + 1
        self.data_manager.move_S_to_T_s()
        self.logger.info(f"Time spend: {self.time_spend:.4f}")
        self.time_spend = 0

        self.logger.info(f"Decision rate: {np.mean(self.decisions):.4f}")
        self.decisions = []
        self.recent_importance = []


class MaxPostMethod(Method):

    def __init__(self, model_handler: ViTHandler, data_manager: DataPoolManager, args, logger, num_classes) -> None:
        super().__init__(model_handler, data_manager, args)
        self.time_spend = 0
        self.num_classes = num_classes
        self.logger = logger
        self.selection_round = 0
        self.selection_info = []
        self.decisions = []
        self.max_post_prob = []

    def select_examples(self) -> List[int]:
        current_time = time.time()
        selected_indices = []

        while len(selected_indices) < self.args.delta:
            if self.data_manager.get_pool_size() == 0:
                break

            batch_data = next(self.data_loader)
            batch_indices = [int(item) for item in batch_data[0]]
            inputs = torch.stack([item for item in batch_data[1]]).to(self.model_handler.device)  # Shape: [batch_size, C, H, W]
            labels = torch.tensor([item for item in batch_data[2]]).to(self.model_handler.device)  # Shape: [batch_size]


            self.model_handler.model.eval()
            with torch.no_grad():
                penultimate = self.model_handler.model.get_penultimate(inputs).detach()
                logits = self.model_handler.model.head(penultimate)
                softmax = F.softmax(logits, dim=1)

            for i, batch_idx in enumerate(batch_indices):
                if (batch_idx in self.data_manager.pool_indices) and (batch_idx not in selected_indices):
                    select, importance = self.check_rule(softmax[i], labels[i].item())
                    if select:
                        selected_indices.append(batch_idx)
                    self.decisions.append(select)
                    self.selection_info.append((self.selection_round, batch_idx, self.decisions[-1],
                                                softmax[i].cpu().numpy() , labels[i].item(), importance))
                    
                
                if len(selected_indices) >= self.args.delta:
                    break
        

        self.data_manager.add_to_S(selected_indices)
        self.time_spend += time.time() - current_time
        return selected_indices

    def check_rule(self, softmax, label):
        if softmax.argmax().item() != label:
            score = 1 - softmax.max().item()
        else:
            score = 0
        self.max_post_prob.append(score)
        select = 100.0 - self.args.selection_p <= percentileofscore(self.max_post_prob, score)
        return select, score

    
    def sleep(self) -> None:
        self.update_S_counts()
        self.selection_round = self.selection_round + 1
        self.data_manager.move_S_to_T_s()
        self.logger.info(f"Time spend: {self.time_spend:.4f}")
        self.time_spend = 0

        self.logger.info(f"Decision rate: {np.mean(self.decisions):.4f}")
        self.decisions = []
        self.max_post_prob = []