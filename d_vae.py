from dall_e import load_model
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dall_e.utils import map_pixels
from torchvision import transforms

def adapt_images(images):
    """Adapt images to the format used by the model.
    Based on https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BEiT/Understanding_BeitForMaskedImageModeling.ipynb (from https://github.com/microsoft/unilm/issues/1235).
    
    Parameters
    ----------
    images : torch.Tensor
        Shape `(n_samples, in_channels, img_size, img_size)`.
    
    Returns
    -------
    torch.Tensor
        Shape `(n_samples, img_size, img_size, in_channels)`.
    """
    visual_token_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((112,112), transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                map_pixels,
            ])
    pixel_values_dall_e = visual_token_transform(images)
    print("Adapted images tensor shape: ", pixel_values_dall_e.shape)
    return pixel_values_dall_e

class DallE_VAE(nn.Module):
    """Dall-E VAE model.
    
    Parameters
    ----------
    image_size : int
        Size of the input for the d-VAE.
    model_dir : str
        Path to the directory with the model weights.
    device : str
        Device to use.
        
    Attributes
    ----------
    encoder
        Encoder model.
    decoder
        Decoder model.
    """
    def __init__(self, image_size=112, model_dir=None, device='cuda'):
        super().__init__()
        self.encoder = None
        self.decoder = None
        self.image_size = image_size

    def load_model(self, model_dir, device):
        """Load model weights.
        
        Parameters
        ----------
        model_dir : str
            Path to the directory with the model weights.
        device : str
            Device to use.
        """
        self.encoder = load_model(os.path.join(model_dir, 'encoder.pkl'), device)
        self.decoder = load_model(os.path.join(model_dir, 'decoder.pkl'), device)
    
    def decode(self, image_seq):
        """Decode image sequence.
        
        Parameters
        ----------
        image_seq : torch.Tensor
            Shape `(n_samples, n_patches, embed_dim)`.
        
        Returns
        -------
        torch.Tensor
            Shape `(n_samples, in_channels, img_size, img_size)`.
        """
        bsz = image_seq.size()[0]
        image_seq = image_seq.view(bsz, self.image_size // 8, self.image_size // 8)
        z = F.one_hot(image_seq, num_classes=self.encoder.vocab_size).permute(0, 3, 1, 2).float()
        return self.decoder(z).float()
    
    def get_codebook_indices(self, images):
        """Get codebook indices for image sequence.	

        Parameters
        ----------
        images : torch.Tensor
            Shape `(n_samples, in_channels, img_size, img_size)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches)`.
        """	
        z_logits = self.encoder(images)
        #print(z_logits.shape)
        return torch.argmax(z_logits, axis=1)
    
    def get_codebook_probs(self, images):
        """Get codebook probabilities for image sequence.	

        Parameters
        ----------
        images : torch.Tensor
            Shape `(n_samples, in_channels, img_size, img_size)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches, vocab_size)`.
        """	
        z_logits = self.encoder(images)
        return F.softmax(z_logits, dim=1)
    
    def forward(self, image_seq_prob, no_process = False):
        """Run forward pass.
        Parameters
        ----------
        image_seq_prob : torch.Tensor
            Shape `(n_samples, n_patches, vocab_size)`.
        Returns
        -------
        torch.Tensor
            Shape `(n_samples, in_channels, img_size, img_size)`.
        """
        if no_process:
            return self.decoder(image_seq_prob.float()).float()
        else:
            bsz, seq_len, num_classes = image_seq_prob.size()
            z = image_seq_prob.view(bsz, self.image_size // 8, self.image_size // 8, self.encoder.vocab_size)
            return self.decoder(z.permute(0, 3, 1, 2).float()).float()
