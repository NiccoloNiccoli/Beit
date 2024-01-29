import torch
from torchvision.models import vit_b_16
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from d_vae import DallE_VAE
from dataset import Dataset
from vision_transformer import VisionTransformer    
import torch.nn.functional as F
import torchvision.datasets as datasets
from PIL import Image

from torch.cuda.amp import autocast, GradScaler

from dall_e.utils import map_pixels
import pickle

#Experiment 1: compare the custom ViT with the torchvision ViT on a classification task

class CifarDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, transformVae):
        self.cifar_dataset = datasets.CIFAR10(root=root, download=True)
        self.transform = transform
        self.transformVae = transformVae
    
    def __len__(self):
        return len(self.cifar_dataset)
    
    def __getitem__(self, idx):
        original_image, label = self.cifar_dataset[idx]
        if self.transform is not None:
            augmented_image1 = self.transform(original_image)
            augmented_image2 = self.transformVae(original_image)
        return augmented_image1, augmented_image2, label

if __name__ == "__main__":
    writer = SummaryWriter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    resize = 224
    image_size = 64
    vt_image_size = 64
    base_transform = transforms.Compose([
        transforms.Resize(resize),
        #transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    small_transform = transforms.Compose([
        #transforms.Resize(resize),
        #transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    cifar_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    dae_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        map_pixels,
    ])
        
    #train_dataset = Dataset(root='tiny-imagenet-200/train', full_size_transform=base_transform, half_size_transform=small_transform)
    train_dataset = CifarDataset(root='cifar10',transform=cifar_transform, transformVae=dae_transform)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, drop_last=True, num_workers=4)

    #validation_dataset = Dataset(root='tiny-imagenet-200/val', full_size_transform=base_transform, half_size_transform=small_transform)
    #validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=4)

    return_all_tokens = False

    model = VisionTransformer(img_size = image_size, return_all_tokens=return_all_tokens, patch_size=8, embed_dim=192, depth = 4).to(device)
    #model.load_state_dict(torch.load("model_70.pth"))
    #model.head = torch.nn.Linear(192, 10).to(device)

    d_vae = DallE_VAE(vt_image_size).to(device)
    d_vae.load_model("dall_e/models/", device)


    #model_torchvision = vit_b_16(num_classes=200).to(device)

    print("Device: ", device)
    print("Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    learning_rate = 1.5e-4
    epochs = 100
    warmup_epochs = 4  # Number of warmup epochs
    #optimizer_torchvision = torch.optim.Adam(model_torchvision.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.05)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.05, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-warmup_epochs, eta_min=0)
    #criterion = torch.nn.CrossEntropyLoss()
    criterion = F.cross_entropy

    scaler = GradScaler()

    for epoch in range(0, epochs):
        #model_torchvision.train()
        model.train()
        running_loss = 0.0
        for i, (img_full_size, img_half_size, _) in enumerate(train_loader):
            with autocast():
                optimizer.zero_grad()
                #outputs = model_torchvision(img_full_size.to(device))
                outputs, mask = model(img_half_size.to(device))
                #print("outputs shape: ", outputs.shape)
                #print("labels shape: ", labels.shape)
            with torch.no_grad():
                tokens = d_vae.get_codebook_indices(img_half_size.to(device)).flatten(1)
                tokens = tokens[mask]
                if return_all_tokens:
                    tokens = tokens.flatten()

            with autocast():
                loss = criterion(outputs.view(-1, 8192), tokens)
            #print("output[0][0]", outputs[0][0])
            #print("tokens shape: ", tokens.shape)
            #outputs = outputs.view(-1, 8192)
            #loss = criterion(outputs, tokens)
            #loss.backward()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if i % 25 == 0:
                print(f"Epoch {epoch}, step {i}, loss {loss.item()}")
                writer.add_scalar("Training Loss (MIM - custom ViT - CIFAR10)", loss.item(), epoch * len(train_loader) + i)
        torch.save(model.state_dict(), f"CIFAR10_custom_vit_{epoch}.pth")
            #running_loss += loss.item()
            #if i % 10 == 0:
                #print(f"Epoch {epoch}, step {i}, loss {loss.item()}")
                #writer.add_scalar("Training Loss", loss.item(), epoch * len(train_loader) + i)
           # writer.add_scalar("Training Loss (1 point per epoch)", running_loss/(i+1), epoch)
        print("output.shape: ", outputs.shape)
        print("mask shape: ", mask.shape)
        if epoch < warmup_epochs:
            lr = learning_rate * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            scheduler.step()
        #writer.flush()
    #writer.close()
    torch.cuda.empty_cache()
