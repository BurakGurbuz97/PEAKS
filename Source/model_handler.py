import torch
import torch.nn as nn
import timm
from typing import Tuple, List
from torchinfo import summary
from Source.vit_model import VisionTransformer
import numpy as np

VIT_16_BASE_CONFIG = {
        "img_size": 224,
        "in_chans": 3,
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "qkv_bias": True,
        "mlp_ratio": 4,
}


VIT_16_SMALL_CONFIG = {
        "img_size": 224,
        "in_chans": 3,
        "patch_size": 16,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "qkv_bias": True,
        "mlp_ratio": 4,
}

def get_config(config_name: str):
    if config_name == "VIT_16_BASE_CONFIG":
        return VIT_16_BASE_CONFIG
    elif config_name == "VIT_16_SMALL_CONFIG":
        return VIT_16_SMALL_CONFIG
    else:
        raise ValueError(f"Invalid config name: {config_name}")

# Helpers
def get_n_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def assert_tensors_equal(t1, t2, atol = 1e-5, rtol = 1e-5):
    a1, a2 = t1.detach().numpy(), t2.detach().numpy()

    np.testing.assert_allclose(a1, a2, rtol=rtol, atol=atol)

def read_freeze_config(freeze_config_path: str) -> List[str]:
    """Read the freeze configuration file and return a list of parameter names to freeze."""
    with open(freeze_config_path, 'r') as f:
        lines = f.readlines()
    # Remove whitespace and empty lines
    param_names = [line.strip() for line in lines if line.strip()]
    return param_names

def freeze_parameters(model: nn.Module, param_names_to_freeze: List[str]) -> None:
    """Freeze parameters in the model based on the provided list of parameter names."""
    model_param_dict = dict(model.named_parameters())
    for name in param_names_to_freeze:
        if name in model_param_dict:
            param = model_param_dict[name]
            param.requires_grad = False
        else:
            print(f"Warning: Parameter '{name}' not found in the model.")

def print_frozen_parameters(model: nn.Module) -> None:
    """Print a list of frozen parameters in the model."""
    frozen_params = [name for name, param in model.named_parameters() if not param.requires_grad]
    print("Frozen parameters:")
    for name in frozen_params:
        print(f" - {name}")


class ViTHandler:
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool = True,
        device: str = 'cuda',
        freeze_config_path: str = '',
        base_vit_config = VIT_16_BASE_CONFIG
    ) -> None:
        """Initialize ViT model handler."""
        self.device = device
        if pretrained:
            model_official = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes
            )

            # Initialize custom model
            self.model = VisionTransformer(num_classes=num_classes, **get_config(base_vit_config))
            for (_, p_o), (_, p_c) in zip(model_official.named_parameters(), self.model.named_parameters()):
                assert p_o.numel() == p_c.numel()
                p_c.data[:] = p_o.data
                assert_tensors_equal(p_c.data, p_o.data)


            inp = torch.rand(1, 3, 224, 224)
            res_c = self.model(inp)
            res_o = model_official(inp)

            # Asserts
            assert get_n_params(self.model) == get_n_params(model_official)
            assert_tensors_equal(res_c, res_o, atol=0.002, rtol=0.002)
            print("All weight loading tests passed!")
            del model_official

            # Freeze specified parameters if freeze_config_path is provided
            if freeze_config_path:
                param_names_to_freeze = read_freeze_config(freeze_config_path)
                freeze_parameters(self.model, param_names_to_freeze)

            print_frozen_parameters(self.model)
            self.model = self.model.to(self.device)
        else:
            self.model = VisionTransformer(num_classes=num_classes, **get_config(base_vit_config)).to(self.device)
            
    def train_batch(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer
    ) -> Tuple[float, float]:
        """Train on a single batch.

        Args:
            batch: Tuple containing images and labels.
            criterion: Loss function.
            optimizer: Optimizer.

        Returns:
            Tuple containing loss and accuracy.
        """
        self.model.train()
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)

        optimizer.zero_grad()
        outputs = self.model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        acc = (outputs.argmax(dim=1) == labels).float().mean().item()
        return loss.item(), acc
            
    def test_batch(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Evaluate on a single batch.

        Args:
            batch: Tuple containing images and labels.
            criterion: Loss function.

        Returns:
            Tuple containing loss and accuracy.
        """
        self.model.eval()
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)

        with torch.no_grad():
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            acc = (outputs.argmax(dim=1) == labels).float().mean().item()

        return loss.item(), acc

    def predict_batch(
        self,
        images: torch.Tensor
    ) -> torch.Tensor:
        """Predict probabilities for a batch of images.

        Args:
            images: Tensor of images.

        Returns:
            Tensor of predicted probabilities.
        """
        self.model.eval()
        images = images.to(self.device)

        with torch.no_grad():
            outputs = self.model(images)
            probs = torch.softmax(outputs, dim=1)

        return probs

    def get_model_summary(self) -> str:
        """Get a summary of the model architecture.

        Returns:
            String representation of the model summary.
        """
        return str(summary(self.model, input_size=(1, 3, 224, 224)))

