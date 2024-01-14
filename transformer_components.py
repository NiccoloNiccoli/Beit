import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """Split image into patches and then embed them.

    Parameters
    ----------
    img_size : int
        Size of the image.

    patch_size : int
        Size of the patch.

    in_channels : int
        Number of input channels.

    embed_dim : int
        Embedding dimension.

    Attributes
    ----------
    n_patches : int
        Number of patches inside the image.
    
    proj : nn.Conv2d
        Convolutional layer that does both the splitting into patches and their embedding.
    """

    def __init__(self, img_size, patch_size, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        #number of patches per image
        self.n_patches = (self.img_size // self.patch_size) ** 2
        #convolutional layer that does both the splitting into patches and their embedding
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, in_channels, img_size, img_size)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches, embed_dim)`.
        """
        x = self.proj(x)
        x = x.flatten(2) #(n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2)

        return x
    
class Attention(nn.Module):
    """Attention mechanism.
    
    Parameters
    ----------
    dim : int
        Input and output dimension of per token features (columns of matrix).
    n_heads : int
        Number of attention heads.
    qkv_bias : bool
        If True then we include bias to the query, key and value projections.
    attn_p : float
        Dropout probability applied to the query, key and value tensors.
    proj_p : float
        Dropout probability applied to the output tensor.

    Attributes
    ----------
    scale : float
        Normalizing constant for the dot product.
    qkv : nn.Linear
        Linear projection for the query, key and value.
    proj : nn.Linear
        Linear mapping that takes in the concatenated output of all attention heads and maps it into a new space.
    attn_drop, proj_drop : nn.Dropout
        Dropout layers.
    """

    def __init__(self, dim, n_heads = 12, qkv_bias = False, attn_p = 0., proj_p = 0.):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias = qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        """ Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        
        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        """
        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError
        
        qkv = self.qkv(x) #(n_samples, n_patches + 1, 3 * dim)
        qkv = qkv.reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim) #(n_samples, n_patches + 1, 3, n_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) #(3, n_samples, n_heads, n_patches + 1, head_dim)
        q, k ,v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1) #(n_samples, n_heads, head_dim, n_patches + 1)
        #dp = (q @ k_t) * self.scale
        q = q * self.scale
        dp = q @ k_t #(n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = dp.softmax(dim = -1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v #(n_samples, n_heads, n_patches + 1, head_dim)
        weighted_avg = weighted_avg.transpose(1, 2) #(n_samples, n_patches + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2) #(n_samples, n_patches + 1, dim)

        x = self.proj(weighted_avg) #(n_samples, n_patches + 1, dim)
        x = self.proj_drop(x)
        return x
    
class MLP(nn.Module):
    """Multilayer perceptron.
    
    Parameters
    ----------
    in_features : int
        Number of input features.
    hidden_features : int
        Number of nodes in the hidden layer.
    out_features : int
        Number of output features.
    p : float
        Dropout probability.

    Attributes
    ----------
    fc : nn.Linear
        The first linear layer.
    act : nn.GELU
        GELU activation function.
    fc2 : nn.Linear
        The second linear layer.
    dropout : nn.Dropout
        Dropout layer.
    """

    #write the init for this class
    def __init__(self, in_features, hidden_features, out_features, p = 0.):
        super().__init__()
        self.fc = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, in_features)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches + 1, out_features)`.
        """
        x = self.fc(x) #(n_samples, n_patches + 1, hidden_features)
        x = self.act(x) #(n_samples, n_patches + 1, hidden_features)
        x = self.dropout(x) #(n_samples, n_patches + 1, hidden_features)
        x = self.fc2(x) #(n_samples, n_patches + 1, out_features)
        x = self.dropout(x) #(n_samples, n_patches + 1, out_features)

        return x
    
class Block(nn.Module):
    """Transformer block.
     
    Parameters
    ----------
    dim : int
        Embedding dimension.
    n_heads : int
        Number of attention heads.
    mlp_ratio : float
        Determines the hidden dimension size of the `MLP` module with respect to `dim`.
    qkv_bias : bool
        If True then we include bias to the query, key and value projections.
    p, attn_p : float
        Dropout probability.
    stochastic_depth : float
        Probability of dropping a residual branch.
    Attributes
    ----------
    norm1, norm2 : nn.LayerNorm
        Layer normalization.
    attn : Attention
        Attention module.
    mlp : MLP
        MLP module.
    """

    #write the init function for this class
    def __init__(self, dim, n_heads, mlp_ratio = 4., qkv_bias = True, p = 0., attn_p = 0., stochastic_depth = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps = 1e-6)
        self.attn = Attention(dim, 
                              n_heads = n_heads, 
                              qkv_bias = qkv_bias, 
                              attn_p = attn_p, 
                              proj_p = p
                              )
        self.norm2 = nn.LayerNorm(dim, eps = 1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(in_features = dim,
                       hidden_features = hidden_features,
                       out_features = dim,
                       )
        self.stochastic_depth = stochastic_depth
        
    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        """
        if self.training and torch.rand(1) > self.stochastic_depth:
            x = x + self.attn(self.norm1(x))

        if self.training and torch.rand(1) > self.stochastic_depth:
            x = x + self.mlp(self.norm2(x))      
                  
        return x