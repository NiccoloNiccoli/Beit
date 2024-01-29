import torch
from torchvision.models import vit_b_16
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import Dataset
from vision_transformer import VisionTransformer    
import torch.nn.functional as F
import torchvision.datasets as datasets

from torch.cuda.amp import autocast, GradScaler

#Experiment 1: compare the custom ViT with the torchvision ViT on a classification task


if __name__ == "__main__":
    torch.cuda.empty_cache()
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
        #transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    #train_dataset = Dataset(root='tiny-imagenet-200/train', full_size_transform=base_transform, half_size_transform=small_transform)
    train_dataset = datasets.CIFAR10(root='cifar10', train=True, download=True, transform=cifar_transform)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=True, num_workers=4)

    #validation_dataset = Dataset(root='tiny-imagenet-200/val', full_size_transform=base_transform, half_size_transform=small_transform)
    #validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=4)

    return_all_tokens = True

    model = VisionTransformer(img_size = image_size, return_all_tokens=return_all_tokens, patch_size=8, embed_dim=192, depth = 8).to(device)
    #model.load_state_dict(torch.load("model_10.pth"))

    #model = vit_b_16(num_classes=10).to(device)
    #model.load_state_dict(torch.load("vit/tv_vit_9.pth"))

    print("Device: ", device)
    print("Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    learning_rate = 1.5e-4
    epochs = 100
    warmup_epochs = 4  # Number of warmup epochs
    #optimizer_torchvision = torch.optim.Adam(model_torchvision.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.05)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.05, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-warmup_epochs, eta_min=0)
    criterion = F.cross_entropy

    scaler = GradScaler()

    for epoch in range(0, epochs):
        #model_torchvision.train()
        model.train()
        running_loss = 0.0
        for i, (data, labels) in enumerate(train_loader):
            with autocast():
                optimizer.zero_grad()
                #outputs = model_torchvision(img_full_size.to(device))
                outputs, _ = model(data.to(device))
                #outputs = torch.argmax(outputs, dim=1)
                #print("outputs shape: ", outputs.shape)
                #print("labels shape: ", labels.shape)
                loss = criterion(outputs, labels.to(device))
            #print("mask shape: ", mask.shape)
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
                writer.add_scalar("Training Loss (ViT training on CIFAR10)", loss.item(), epoch * len(train_loader) + i)
            #running_loss += loss.item()
            #if i % 10 == 0:
                #print(f"Epoch {epoch}, step {i}, loss {loss.item()}")
                #writer.add_scalar("Training Loss", loss.item(), epoch * len(train_loader) + i)
           # writer.add_scalar("Training Loss (1 point per epoch)", running_loss/(i+1), epoch)
        
        if epoch < warmup_epochs:
            lr = learning_rate * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            torch.save(model.state_dict(), f"vit/CIFAR10/custom_vit_{epoch}.pth")
            scheduler.step()
        #writer.flush()
    #writer.close()
    torch.cuda.empty_cache()
