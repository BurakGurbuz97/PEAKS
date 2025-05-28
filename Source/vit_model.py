import torch
import torch.nn as nn
from timm.models.layers import DropPath
from typing import Optional

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) * (img_size // patch_size)
        # Use a conv layer to extract patches and perform embedding
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, in_chans, img_size, img_size)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches, embed_dim)`.
        """
        x = self.proj(x) # (n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2).transpose(1, 2) # (n_samples, n_patches, embed_dim)
        return x
    

    

class Attention(nn.Module):
    """ Multi-head self attention layer """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 12,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Define learnable parameters Q, K, V
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + cls, embed_dim)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches + cls, embed_dim)`.
        """
        n_samples, n_patches, embed_dim = x.shape
        qkv = self.qkv(x)  # (n_samples, n_patches + 1, 3 * embed_dim)
        qkv = qkv.reshape(n_samples, n_patches, 3, self.num_heads, self.head_dim)  # (n_samples, n_patches + cls, 3, n_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, n_samples, n_heads, n_patches + cls, head_dim)
        q, k, v = qkv.unbind(0)  # Each of shape: [n_samples, num_heads, n_patches + cls, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # Shape: [n_samples, num_heads, n_patches + cls, n_patches + cls]
        attn = attn.softmax(dim=-1)  # Softmax over the last dimension
        attn = self.attn_drop(attn)

        weighted_avg_values = (attn @ v)  # Shape: [n_samples, num_heads, n_patches + cls, head_dim]
        weighted_avg_values = weighted_avg_values.transpose(1, 2)  # Shape: [n_samples, n_patches + cls, num_heads, head_dim]
        weighted_avg_values = weighted_avg_values.flatten(2)  # Shape: [n_samples, n_patches + cls, embed_dim]

        x = self.proj(weighted_avg_values)  # Shape: [n_samples, n_patches + cls, embed_dim]
        x = self.proj_drop(x)

        return x
    

class Mlp(nn.Module):
    """ Multilayer perceptron """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop_p: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or int(in_features * 4)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, in_features)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches +1, out_features)`
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    """ Transformer block """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_p: float = 0.0,
        attn_drop_p: float = 0.0,
        drop_path_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = Attention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop_p,
            proj_drop=drop_p,
        )
        self.drop_path = DropPath(drop_path_p) if drop_path_p > 0.0 else nn.Identity() # Stochastic Depth - 0 by default (based on timm implementation)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=mlp_hidden_dim, drop_p=drop_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + cls, embed_dim)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches + cls, embed_dim)`.
        """
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    

class VisionTransformer(nn.Module):
    """ Vision Transformer """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim 

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_p=drop_rate,
                    attn_drop_p=attn_drop_rate,
                    drop_path_p=drop_path_rate,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, in_chans, img_size, img_size)`.

        Returns
        -------
        torch.Tensor
            logits of shape `(n_samples, num_classes)`.
        """
        n_samples = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(n_samples, -1, -1) # (n_samples, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1) #  (n_samples, 1 + cls, embed_dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)

        return x

    def get_penultimate(self, x: torch.Tensor) -> torch.Tensor:
        n_samples = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(n_samples, -1, -1) # (n_samples, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1) #  (n_samples, 1 + cls, embed_dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
            
        return x[:, 0]
