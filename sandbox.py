import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from masking_generator import MaskingGenerator
from vision_transformer import VisionTransformer
from d_vae import DallE_VAE, adapt_images
from modeling_discrete_vae import Dalle_VAE as DVAE
from dataset import Dataset
from dall_e.utils import map_pixels
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
        b, c, h, w = x.shape
        assert h == self.img_size and w == self.img_size, \
            f"Input image size ({h}*{w}) doesn't match model ({self.img_size}*{self.img_size})."
        
        x = self.proj(x)
        x = x.flatten(2) #(n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2)

        return x
    

if __name__ == "__mains__":
    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
    visual_token_transform = transforms.Compose([
        transforms.Resize((112,112), transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        map_pixels,
        ])
    train_dataset = Dataset(root='E:/Download/imagenette2-320/imagenette2-320/train', full_size_transform=transform, half_size_transform=visual_token_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=4)
    for f_data, h_data, target in train_loader:
        print("(full size) data.shape: ", f_data.shape)
        print("(half size) data.shape: ", h_data.shape)
        print("target length", len(target))
        break

    if False:
        model = PatchEmbed(img_size=224, patch_size=16)
        #plot a random image from the dataloader
        plt.imshow(data[0].permute(1, 2, 0))
        plt.show()

        output = model(data)
        print("output: ", output.shape)

        mask = MaskingGenerator((14,14))
        mask = mask.generate_mask().flatten()
        print("mask: ", mask.shape)

        #mask is a tensor and I want to make a batch of masks
        batch_of_masks = mask.unsqueeze(0)
        batch_of_masks = batch_of_masks.repeat(32, 1)
        batch_of_masks = batch_of_masks.unsqueeze(-1)
        print("batch_of_masks: ", batch_of_masks.shape)

        output_masked = output * ~batch_of_masks
        print("output_masked: ", output_masked.shape) #(batch_size, n_patches, embed_dim)
        ou = output_masked[0][1]
        ou = ou.view(16, 16, 3).detach().numpy()
        fig, axes = plt.subplots(14, 14, figsize=(14*2, 14*2))
        patches = output_masked[0]
        n_patches = patches.shape[0]
        # Iterate over the patches and plot each one
        for i, ax in enumerate(axes.flatten()):
            if i < n_patches:
                # Reshape the patch to its original shape and detach it from gradients
                patch = patches[i].view(16, 16, 3).detach().cpu().numpy()
                
                # Plot the patch
                ax.imshow(patch)
                ax.axis('off')
            else:
                # If there are more subplots than patches, hide the extra subplots
                ax.axis('off')

        # Show the plot
        plt.show() 
    
    if True:
        model = VisionTransformer(return_all_tokens=False).to("cuda")
        outputs, mask = model(f_data.to("cuda"))
        print("output: ", outputs.shape)
        print("mask: ", mask.shape)
        #outputs = torch.argmax(outputs, axis=2)
        print("outputs: ", outputs.shape, outputs.dtype)
        #print(" i 000",outputs[0])
    #dvae = DallE_VAE(model_dir="dall_e/models/", device="cuda")
    dvae = DallE_VAE(112)
    dvae.load_model(model_dir="dall_e/models/", device="cuda")
    input_ids = dvae.get_codebook_indices(h_data.to("cuda")).flatten(1)
    #input_ids = F.one_hot(input_ids, num_classes=8192).permute(0, 3, 1, 2).float()
    print("input_ids: ", input_ids.shape, input_ids.dtype)
    #print(" i 000",input_ids[0])
    labels = input_ids[mask]
    print(labels.shape)

    loss = nn.CrossEntropyLoss()(input = outputs.view(-1, 8192), target = labels)
    print("loss: ", loss.item())


