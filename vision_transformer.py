import torch
import torch.nn as nn
import math
from transformer_components import PatchEmbed, Block
from masking_generator import MaskingGenerator

 
class VisionTransformer(nn.Module):
    """ Vision transformer for MIM.
    Parameters
    ----------
    img_size : int
        Both height and the width of the image (it is a square).
    patch_size : int
        Both height and the width of the patch (it is a square).
    in_channels : int
        Number of input color channels.
    vocab_size : int
        Vocabulary size of the elements of the predicted tokens.
    embed_dim : int
        Dimensionality of the token/patch embeddings.
    depth : int
        Number of blocks.
    n_heads : int
        Number of attention heads.
    mlp_ratio : float
        Determines the hidden dimension size of the `MLP` module with respect to `embed_dim`.
    qkv_bias : bool
        If True then we include bias to the query, key and value projections.
    p, attn_p : float
        Dropout probability.
    init_value : float
        Tokens initialization value.
    return_all_tokens : bool
        Whether to return all tokens or just the masked ones. 
    Attributes
    ----------
    patch_embed : PatchEmbed
        Instance of `PatchEmbed` layer.
    masking_generator : MaskingGenerator
        Instance of `MaskingGenerator` layer.
    mask_token : nn.Parameter
        Learnable parameter that will represent the masked tokens.
    cls_token : nn.Parameter
        Learnable parameter that will represent the first token in the sequence.
    pos_emb : nn.Parameter
        Positional embeddings.
    pos_drop : nn.Dropout
        Dropout layer.
    blocks : nn.ModuleList
        List of `Block` modules.
    norm : nn.LayerNorm
        Layer normalization.
    """
    #write the init function for this class
    def __init__(self, img_size = 224, patch_size = 16, in_channels = 3, vocab_size = 8192, embed_dim = 768, depth = 12, n_heads = 12, mlp_ratio = 4., qkv_bias = True, p = 0., attn_p = 0., init_value = 0.02, return_all_tokens = False):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size = img_size, 
            patch_size = patch_size, 
            in_channels = in_channels, 
            embed_dim = embed_dim
            ) 
        self.masking_generator = MaskingGenerator(input_size = img_size // patch_size, min_number_of_patches= 2, max_masking_factor=0.4)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))
        self.pos_drop = nn.Dropout(p = p)
        self.blocks = nn.ModuleList([
            Block(
                dim = embed_dim, 
                n_heads = n_heads, 
                mlp_ratio = mlp_ratio, 
                qkv_bias = qkv_bias, 
                p = p, 
                attn_p = attn_p
                )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps = 1e-6)

        #self.head = nn.Linear(embed_dim, 10)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.Linear(embed_dim * 4, vocab_size)
        )

        nn.init.trunc_normal_(self.pos_embed, std = init_value)
        nn.init.trunc_normal_(self.cls_token, std = init_value)
        nn.init.trunc_normal_(self.mask_token, std = init_value)
        nn.init.trunc_normal_(self.head[0].weight, std = init_value)
        nn.init.trunc_normal_(self.head[1].weight, std = init_value)    
        
        self.return_all_tokens = return_all_tokens

        self.apply(self._init_weights)

        for layer_id, layer in enumerate(self.blocks):
            self.rescale(layer.attn.proj.weight.data, layer_id + 1)
            self.rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std = 0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            nn.init.trunc_normal_(module.weight, std = 0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def rescale(self, param, layer_id):
        param.div_(math.sqrt(2.0 * layer_id))

    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, in_channels, img_size, img_size)`.

        Returns
        -------
        logits : torch.Tensor
            Logits over all classes.
        """

        
        n_samples = x.shape[0]
        x = self.patch_embed(x)
        batch_size, seq_len, embed_dim = x.shape

        if False:
            mask = self.masking_generator.generate_mask().flatten() #(n_patches)
            mask = mask.unsqueeze(0) #(1, n_patches)
            batch_of_masks = mask.repeat(n_samples, 1) #(n_samples, n_patches) #bools

            mask_token = self.mask_token.expand(batch_size, seq_len, -1) #(n_samples, n_patches, embed_dim) #ints

            w = batch_of_masks.unsqueeze(-1).type_as(mask_token) #(n_samples, n_patches, 1) #ints
            #print("w.shape: ", w.shape, w.dtype)       
            x = x * (1 - w) + w * mask_token #(n_samples, n_patches, embed_dim)
        else:
            batch_of_masks = None
        cls_token = self.cls_token.expand(n_samples, -1, -1) #(n_samples, 1, embed_dim)
        x = torch.cat((cls_token, x), dim = 1) #(n_samples, 1 + n_patches, embed_dim)
        x = x + self.pos_embed #(n_samples, 1 + n_patches, embed_dim)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        if False:
            x = x[:, 1:] #(n_samples, n_patches, embed_dim)
            #print("x.shape: ", x.shape, x.dtype)
        else:
            x = x[:, 0] #(n_samples, embed_dim)

        if self.return_all_tokens:
            x = self.head(x) #(n_samples, n_patches, vocab_size)
            return x, batch_of_masks
        else:
            batch_of_masks = batch_of_masks.to("cuda")
            x = x[batch_of_masks.unsqueeze(-1).expand_as(x)] #(n_samples * n_masked_patches * embed_dim)
            #print("x.shape: ", x.shape, x.dtype)
            x = x.view(n_samples, -1, embed_dim)
            #print("x.shape: ", x.shape, x.dtype)
            x = self.head(x) #(n_samples, n_patches, vocab_size)
            return x, batch_of_masks