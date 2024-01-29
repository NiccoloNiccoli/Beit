from collections import OrderedDict
import math
from torchvision.models import VisionTransformer
from torchvision.ops.misc import MLP
import torch.nn as nn
import torch

from relative_position_bias import RelativePositionBias
from typing import Any, Callable, Dict, List, NamedTuple, Optional
from functools import partial

class MLPBlock(MLP):
    """Transformer MLP block."""
    #copied from torchvision repo because it is not importable :<

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
class TorchvisionEncoderBlock(nn.Module):
    """Transformer encoder block."""
    #copied from torchvision repo because it is not importable :<
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y
class TorchvisionEncoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""
    #copied from torchvision repo because it is not importable :<

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = TorchvisionEncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor
    
class EBlock(TorchvisionEncoderBlock):
    def __init__(self, num_head:int, hidden_dim:int, mlp_dim:int, dropout:float = 0.0, attention_dropout:float = 0.0, drop_path_rate:float = 0.1, init_values = 0.1):
        super().__init__(num_head, hidden_dim, mlp_dim, dropout, attention_dropout)
        if drop_path_rate > 0.0:
            self.drop_path = DropPath(drop_path_rate)
        else:
            self.drop_path = nn.Identity()

        if init_values is not None:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(hidden_dim), requires_grad = True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(hidden_dim), requires_grad = True)

    def forward(self, x : torch.Tensor, relative_position_bias = None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.self_attention(self.ln_1(x), self.ln_1(x), self.ln_1(x))[0]) #fixme introdurre relative_position_bias
            x = x + self.drop_path(self.mlp(self.ln_2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.self_attention(self.ln_1(x), self.ln_1(x), self.ln_1(x), need_weights=False)[0])
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.ln_2(x)))
        return x
    
class Enc(TorchvisionEncoder):
    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        drop_path:list,
        init_values: float = 0.1,
    ):
        super().__init__(seq_length, num_layers, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        
        for i in range(num_layers):
            
            layers[f"encoder_layer_{i}"] = EBlock(num_heads, hidden_dim, mlp_dim, dropout, attention_dropout, drop_path[i], init_values)

        self.layers = nn.Sequential(layers)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))
    
class ViT(VisionTransformer):
    def __init__(
        self,
        vocab_size: int,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        return_all_tokens: bool = False,
        drop_path_rate: float = 0.1,
    ):
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            num_classes=num_classes,
        )
        self.return_all_tokens = return_all_tokens
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.relative_position_bias = RelativePositionBias(window_size = (image_size // patch_size, image_size // patch_size), num_heads = num_heads)

        stochastic_depth_decay_rule = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]

        self.encoder = Enc(
            seq_length = self.seq_length,
            num_layers = num_layers,
            num_heads = num_heads,
            hidden_dim = hidden_dim,
            mlp_dim = mlp_dim,
            dropout = dropout,
            attention_dropout = attention_dropout,
            drop_path = stochastic_depth_decay_rule,
        )
        
        self.init_std = 0.02

        self.MIMhead = nn.Linear(hidden_dim, vocab_size)

        torch.nn.init.trunc_normal_(self.class_token, std=self.init_std)
        torch.nn.init.trunc_normal_(self.mask_token, std=self.init_std)
        torch.nn.init.trunc_normal_(self.MIMhead.weight, std=self.init_std)
        self.apply(self._init_weights)
        #self.fix_init_weight() #todo (troppo complicato da modificare)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, std=self.init_std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    #def fix_init_weight(self):
        #def rescale(param, layer_id):
            #param.div_(math.sqrt(2.0 * layer_id))
        
        #for layer_id, layer in enumerate(self.encoder.layers):
            #rescale(layer.attention.proj.weight.data, layer_id + 1)
            #rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def forward(self, x, mask = None):
        # Reshape and permute the input tensor
        x = super()._process_input(x)
        n = x.shape[0]

        #apply mask
        if mask is not None: #fixme
            mask_token = self.mask_token.expand(n, self.seq_length - 1, -1) #(n, n_patches, embed_dim)
            w = mask.unsqueeze(-1).type_as(mask_token)
            x = x * (1 - w) + mask_token * w #(n, n_patches, embed_dim)

        # Expand the class token to the full batch
        batch_class_tokens = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_tokens, x], dim=1)

        x = self.encoder(x)

        if mask is not None:
            x = x[:, 1:]
            if self.return_all_tokens:
                x = self.MIMhead(x)
            else:
                x = self.MIMhead(x[mask])
        else:
            x = x[:, 0]
            x = self.heads(x)
        return x
