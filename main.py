import torch
from d_vae import DallE_VAE
from vision_transformer import VisionTransformer
from torch.nn.functional import cosine_similarity

from dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from dall_e.utils import map_pixels

import time


from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
    writer = SummaryWriter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    resize = 64
    image_size = 64
    vt_image_size = 64
    base_transform = transforms.Compose([
        #transforms.Resize(resize),
        #transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    visual_token_transform = transforms.Compose([
        transforms.Resize((vt_image_size, vt_image_size), transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        map_pixels,
    ])
    train_dataset = Dataset(root='tiny-imagenet-200/train', full_size_transform=base_transform, half_size_transform=visual_token_transform)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, drop_last=True, num_workers=4)

    validation_dataset = Dataset(root='tiny-imagenet-200/val', full_size_transform=base_transform, half_size_transform=visual_token_transform)
    validation_loader = DataLoader(validation_dataset, batch_size=1024, shuffle=True, drop_last=True, num_workers=4)

    return_all_tokens = False

    model = VisionTransformer(img_size = image_size, return_all_tokens=return_all_tokens, patch_size=8, embed_dim=192, depth = 8).to(device)
    d_vae = DallE_VAE(vt_image_size).to(device)
    d_vae.load_model("dall_e/models/", device)

    print("Device: ", device)
    print("Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    learning_rate = 1.5e-4
    #learning_rate = 0.0001
    epochs = 100
    warmup_epochs = 0 
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.05)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.05, momentum=0.9)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    criterion = torch.nn.CrossEntropyLoss()

    tic = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (img_full_size, img_half_size, _) in enumerate(train_loader):
            optimizer.zero_grad()
           
            outputs, mask = model(img_full_size.to(device))
            #print("outputs shape: ", outputs.shape)
            #print("mask shape: ", mask.shape)
            #print("output[0][0]", outputs[0][0])
            with torch.no_grad():
                tokens = d_vae.get_codebook_indices(img_half_size.to(device)).flatten(1)
                tokens = tokens[mask]
                if return_all_tokens:
                    tokens = tokens.flatten()
            #print("tokens shape: ", tokens.shape)
            outputs = outputs.view(-1, 8192)
            loss = criterion(outputs, tokens)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 0:
                print(f"Epoch {epoch}, step {i}, loss {loss.item()}")
                writer.add_scalar("Training Loss", loss.item(), epoch * len(train_loader) + i)
            writer.add_scalar("Training Loss (1 point per epoch)", running_loss/(i+1), epoch)
        
        if epoch < warmup_epochs :
            lr = learning_rate * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            scheduler.step()

        accuracy = (outputs.max(-1)[1] == tokens).float().mean().item()
        writer.add_scalar("Training Accuracy", accuracy, epoch)
        writer.flush()

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"model_{epoch}.pth")
        if False:
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    total = 0
                    correct = 0
                    for i, (img_full_size, img_half_size, _) in enumerate(validation_loader):
                        outputs, mask = model(img_full_size.to(device))
                        tokens = d_vae.get_codebook_indices(img_half_size.to(device)).flatten(1)
                        tokens = tokens[mask]
                        _, predicted = torch.max(outputs.data, 2)
                        total += tokens.size(0)
                        correct += (predicted.flatten() == tokens).sum().item()
                    accuracy = 100 * correct / total
                    print(f"Accuracy on masked image modeling task: {accuracy}%")
                    writer.add_scalar("Validation Accuracy", accuracy, epoch)
    toc = time.time()
    print("Elapsed time: ", toc-tic)
    writer.close()
