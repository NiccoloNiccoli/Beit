import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
from vit import ViT
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = 'Program name',
        description = 'Program description',
        epilog = 'Program epilog'
    )
    parser.add_argument('--pretrained', dest='isPretrained', action='store_true', help='Whether to run pretrained model or not')
    args = parser.parse_args()
    print("Using pretrained model? ", args.isPretrained)


    #------------------------Setting the device------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    #------------------------Setting the seed------------------------
    seed = 28
    torch.manual_seed(seed)
    
    #------------------------Setting the hyperparameters------------------------
    batch_size = 128
    image_size = 224
    patch_size = 16
    num_classes = 100

    learning_rate = 5e-4

    #------------------------Setting the Tensorboard writer------------------------
    comment = f'BATCH_{batch_size}_LR_HEAD_{learning_rate}_LR_BACKBONE_{learning_rate / 100}'
    if args.isPretrained:
        comment = comment + "_PRETRAINED"
    writer = SummaryWriter(comment = comment)

    #------------------------Defining the transform------------------------
    cifar100_transform = transforms.Compose([
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    #------------------------Loading the dataset splits------------------------
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=cifar100_transform)

    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=cifar100_transform)

    #------------------------Defining the dataloaders------------------------
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

    #------------------------Defining the model------------------------
    model = ViT(8192, image_size=image_size, patch_size=patch_size, hidden_dim=768, num_layers=12, num_heads=12, mlp_dim=3072)
    if args.isPretrained:
        checkpoint = torch.load("models/checkpoint_34.pth")
        model.load_state_dict(checkpoint['model'])
    model.heads = nn.Linear(768, num_classes)
    model.to(device)

    #------------------------Defining the optimizer------------------------
    optimizer = optim.Adam([
            {'params': model.conv_proj.parameters(), 'lr': learning_rate / 100},
            {'params': model.encoder.parameters(), 'lr': learning_rate / 100},
            {'params': model.heads.parameters(), 'lr': learning_rate},
    ], betas = (0.9, 0.999))

    #------------------------Defining the loss function------------------------
    criterion = nn.CrossEntropyLoss()

    #------------------------Defining the training loop------------------------
    epochs = 14
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\t'.format(
                    epoch, batch_idx * len(data), len(train_dataloader.dataset),
                    100. * batch_idx / len(train_dataloader), loss.item()))

        #------------------------Defining the validation loop------------------------
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_dataloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(test_dataloader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_dataloader.dataset),
                100. * correct / len(test_dataloader.dataset)))
        writer.add_scalar('[Finetuning] Test accuracy', 100. * correct / len(test_dataloader.dataset), epoch)
    
    #------------------------Saving the model------------------------
    filename = f'models/finetuning/checkpoint_{epoch}'
    if args.isPretrained:
        filename = filename + "_pretrained"
    filename = filename + ".pth"
    torch.save({'model': model.state_dict()}, filename)
    writer.close()
    torch.cuda.empty_cache()



