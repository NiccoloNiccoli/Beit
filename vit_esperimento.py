import torch
from torchvision.models import vit_b_16
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from d_vae import DallE_VAE
from dataset import ImageNetDataset
from masking_generator import MaskingGenerator
from vision_transformer import VisionTransformer    
import torch.nn.functional as F
import torchvision.datasets as datasets
from PIL import Image

from torch.cuda.amp import autocast, GradScaler

from dall_e.utils import map_pixels
import pickle
import numpy as np

from vit import ViT
import torch.nn as nn

from ZZZ_cacap import cosine_scheduler
import warnings
import random
import math
import torchvision.transforms.functional as T_F
#Experiment 1: pretrained model vs non-pretrained one

class Scaler:
    def __init__(self):
        self._scaler = GradScaler()

    def __call__(self, loss, optimizer, clip_grad = None, parameters = None, update_grad = True):
        self._scaler.scale(loss).backward()
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = self.get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def get_grad_norm_(self, parameters, norm_type: float = 2.0) -> torch.Tensor:
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        norm_type = float(norm_type)
        if len(parameters) == 0:
            return torch.tensor(0.)
        device = parameters[0].grad.device
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
        return total_norm
_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}
_RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)
def _pil_interp(method):
    if method == 'bicubic':
        return Image.BICUBIC
    elif method == 'lanczos':
        return Image.LANCZOS
    elif method == 'hamming':
        return Image.HAMMING
    else:
        # default bilinear, do we want to allow nearest?
        return Image.BILINEAR
class RandomResizedCropAndInterpolationWithTwoPic:
    """Crop the given PIL Image to random size and aspect ratio with random interpolation.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, second_size=None, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation='bilinear', second_interpolation='lanczos'):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if second_size is not None:
            if isinstance(second_size, tuple):
                self.second_size = second_size
            else:
                self.second_size = (second_size, second_size)
        else:
            self.second_size = None
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        if interpolation == 'random':
            self.interpolation = _RANDOM_INTERPOLATION
        else:
            self.interpolation = _pil_interp(interpolation)
        self.second_interpolation = _pil_interp(second_interpolation)
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        area = img.size[0] * img.size[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if in_ratio < min(ratio):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        if self.second_size is None:
            return T_F.resized_crop(img, i, j, h, w, self.size, interpolation)
        else:
            return T_F.resized_crop(img, i, j, h, w, self.size, interpolation), \
                   T_F.resized_crop(img, i, j, h, w, self.second_size, self.second_interpolation)

    def __repr__(self):
        if isinstance(self.interpolation, (tuple, list)):
            interpolate_str = ' '.join([_pil_interpolation_to_str[x] for x in self.interpolation])
        else:
            interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0}'.format(interpolate_str)
        if self.second_size is not None:
            format_string += ', second_size={0}'.format(self.second_size)
            format_string += ', second_interpolation={0}'.format(_pil_interpolation_to_str[self.second_interpolation])
        format_string += ')'
        return format_string
    
def get_parameter_groups(model, weight_decay=1e-5, skip_list=()):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.0
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        
        if group_name not in parameter_group_names:
            scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale    
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    return list(parameter_group_vars.values())


if __name__ == "__main__":
    writer = SummaryWriter()
    torch.cuda.empty_cache()
    seed = 28
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

   
    #------------------------Loading the dataset------------------------
    image_size = 224
    small_image_size = 112
    patch_size = 16
    n_patches = image_size // patch_size
    batch_size = 128
    val_batch_size = 128

    base_transform = transforms.Compose([
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(p=0.5),
        RandomResizedCropAndInterpolationWithTwoPic(
            size=image_size, second_size=small_image_size,
            interpolation='bicubic', second_interpolation='lanczos'
        ),
    ])
    patch_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ])
    dvae_transform = transforms.Compose([
        transforms.ToTensor(),
        map_pixels
    ])
    train_dataset = ImageNetDataset(root = "ILSVRC", split='train', base_transform=base_transform, patch_transform = patch_transform, vae_transform=dvae_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=16, pin_memory=True)

    validation_dataset = ImageNetDataset(root = "ILSVRC", split='val', base_transform=base_transform, patch_transform = patch_transform, vae_transform=dvae_transform)
    validation_loader = DataLoader(validation_dataset, batch_size=val_batch_size, shuffle=True, drop_last=True, num_workers=16)

    #------------------------Loading the models------------------------
    vocab_size = 8192
    
    hidden_dim = 768
    mlp_dim = 3072
    n_layers = 12
    n_heads = 12

    

    return_all_tokens = False

    model = ViT(vocab_size=vocab_size,
                image_size=image_size,
                patch_size=patch_size,
                num_layers=n_layers,
                num_heads=n_heads,
                hidden_dim=hidden_dim,
                mlp_dim=mlp_dim,
                return_all_tokens = return_all_tokens).to(device)

    dVae = DallE_VAE().to(device)
    dVae.load_model("dall_e/models/", device)

    #masking_generator = MaskingGenerator(input_size = n_patches, max_masking_factor=0.4)

    print("Device: ", device)
    print("Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    #------------------------Setting the optimizer------------------------
    learning_rate = 5e-4
    epochs = 99
    warmup_epochs = 10  # Number of warmup epochs
    weight_decay = 0.05
    skip = {'pos_embed', 'cls_token'}
    parameters = get_parameter_groups(model, weight_decay, skip)
    weight_decay = 0.05
    optional_args = dict(lr = learning_rate, weight_decay = weight_decay, eps = 1e-10, betas = (0.9, 0.999))

    optimizer = torch.optim.AdamW(parameters, **optional_args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-warmup_epochs, eta_min=0)
    criterion = F.cross_entropy

    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_scheduler, gamma=0.1)
    #print(lr_scheduler)
    scaler = Scaler()

    pipip = True
    popop = True
    
    start_epoch = 0
    steps_between_prints = 32
    lr_warmup = torch.linspace(0, learning_rate, warmup_epochs * len(train_loader))

    

    if True:
        checkpoint = torch.load("models/checkpoint_34.pth")
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_sched'])
        start_epoch = checkpoint['epoch'] + 1
        print("Loaded checkpoint")
    #------------------------Training Loop------------------------
    #------------------------------------------------
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    #------------------------------------------------

    accumulation_steps = 8

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        accum_loss = 0.0
        for i, batch in enumerate(train_loader):
            
            img, img_vae, mask, _ = batch

            if epoch < warmup_epochs:
                lr = lr_warmup[(epoch - start_epoch) * len(train_loader) + i]
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            with torch.no_grad():
                tokens = dVae.get_codebook_indices(img_vae.to(device)).flatten(1)
                mask = mask.flatten(1).to(torch.bool)
                labels = tokens[mask]
                if return_all_tokens: #XXX
                    tokens = tokens.flatten() #XXX
           
            with autocast():
                outputs = model(img.to(device), mask)
                if pipip:
                    print("[TRAIN] outputs shape: ", outputs.shape)
                    print("[TRAIN] labels", labels)
                    pipip = False
                loss = nn.CrossEntropyLoss()(outputs, labels)
            
            total_loss += loss.item()
            
            #using gradient accumulation to simulate a bigger batch size
            if False:
                if i  % accumulation_steps != 0:
                    accum_loss += loss
                elif i  % accumulation_steps == 0:
                    accum_loss += loss
                    accum_loss /= accumulation_steps
                    accum_loss = 0.0
            
            if (i + 1) % accumulation_steps == 0:
                grad_norm = scaler(loss, optimizer, clip_grad = None, parameters = model.parameters(), update_grad=True)  
                optimizer.zero_grad()
            else:
                grad_norm = scaler(loss, optimizer, clip_grad = None, parameters = model.parameters(), update_grad=False)

           
            if (i+1) % steps_between_prints == 0:
                mim_accuracy = (outputs.max(-1)[1] == labels).float().mean().item()
                for param_group in optimizer.param_groups:
                    _lr = param_group['lr']
                    break
                print(f"Epoch {epoch}, step {i}/{len(train_dataset) // batch_size}, loss {loss.item():.5f}, avg loss {(total_loss/(i+1)):.5f} mim accuracy {mim_accuracy:.5f}, grad norm {grad_norm:.5f}, lr {_lr:.5e}")
                writer.add_scalar("Training Loss", loss.item(), epoch * len(train_loader) + i)
                writer.add_scalar("Mean Training Loss", total_loss/(i+1), epoch * len(train_loader) + i)

        if epoch >= warmup_epochs:
            scheduler.step()

        if epoch % 1 == 0:
            checkpoint = { 
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_sched': scheduler.state_dict()}

            torch.save(checkpoint, f'models/checkpoint_{epoch}.pth')
        
        if epoch % 2 == 0:
            #test on validation to see imporvements
            model.eval()
            with torch.no_grad():
                total = 0
                correct = 0
                for i, batch in enumerate(validation_loader):
                    img, _, _, labels = batch
                    outputs = model(img.to(device))
                    if popop:
                        print("[VAL] outputs shape: ", outputs.shape)
                        print("[VAL] labels", labels)
                        popop = False

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels.to(device)).sum().item()
                writer.add_scalar("Validation Accuracy (on a cls task)", 100 * correct / total, epoch)

        writer.flush()
    writer.close()
    torch.cuda.empty_cache()
