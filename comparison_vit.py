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

from vit_esperimento import RandomResizedCropAndInterpolationWithTwoPic, Scaler, get_parameter_groups

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


    print("Device: ", device)
    print("Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    #------------------------Setting the optimizer------------------------
    learning_rate = 5e-6
    epochs = 20
    #warmup_epochs = 10  # Number of warmup epochs
    weight_decay = 0.05
    skip = {'pos_embed', 'cls_token'}
    parameters = get_parameter_groups(model, weight_decay, skip)
    weight_decay = 0.
    optional_args = dict(lr = learning_rate, weight_decay = weight_decay, eps = 1e-14, betas = (0.9, 0.999))

    optimizer = torch.optim.AdamW(parameters, **optional_args)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    criterion = F.cross_entropy

    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_scheduler, gamma=0.1)
    #print(lr_scheduler)
    scaler = Scaler()

    pipip = True
    popop = True
    
    start_epoch = 0
    steps_between_prints = 50

    
    if True:
        #checkpoint = torch.load("models/checkpoint_14.pth")
        #model.load_state_dict(checkpoint['model'])
        #optimizer.load_state_dict(checkpoint['optimizer'])
        #start_epoch = checkpoint['epoch'] + 1
        print("Loaded checkpoint")
    #------------------------Training Loop------------------------
    #------------------------------------------------
    #for layer_id, layer in enumerate(model.encoder.layers):
        #layer.self_attention.attn_output.weight.data.div_(math.sqrt(2.0 * (layer_id + 1.0)))
        #layer.mlp.layers[-1].weight.data.div_(math.sqrt(2.0 * (layer_id + 1.0)))
    #for param_group in optimizer.param_groups:
        #param_group['lr'] = learning_rate
    #scheduler.T_max = epochs - start_epoch
    #------------------------------------------------

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        for i, batch in enumerate(train_loader):
            
            img, _, _, labels = batch
            labels = labels.to(device)
            with autocast():
                outputs = model(img.to(device))
                if pipip:
                    print("[TRAIN] outputs shape: ", outputs.shape)
                    print("[TRAIN] labels", labels)
                    pipip = False
                loss = nn.CrossEntropyLoss()(outputs, labels)
            
            total_loss += loss.item()

            mim_accuracy = (outputs.max(-1)[1] == labels).float().mean().item()

            optimizer.zero_grad()
            grad_norm = scaler(loss, optimizer, clip_grad = None, parameters = model.parameters())

            if i % steps_between_prints == 0:
                for param_group in optimizer.param_groups:
                    _lr = param_group['lr']
                    break
                print(f"Epoch {epoch}, step {i}/{len(train_dataset) // batch_size}, loss {loss.item():.5f}, avg loss {(total_loss/(i+1)):.5f} mim accuracy {mim_accuracy:.5f}, grad norm {grad_norm:.5f}, lr {_lr:.5e}")
                writer.add_scalar("Training Loss [FINETUNING]", loss.item(), epoch * len(train_loader) + i)
                writer.add_scalar("Mean Training Loss [FINETUNING]", total_loss/(i+1), epoch * len(train_loader) + i)

        if epoch % 1 == 0:
            checkpoint = { 
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            }

            torch.save(checkpoint, f'models/finetuning/checkpoint_{epoch}.pth')
        
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
                writer.add_scalar("Validation Accuracy [FINETUNING]", 100 * correct / total, epoch)

        writer.flush()
    writer.close()
    torch.cuda.empty_cache()