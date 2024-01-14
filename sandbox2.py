import torch
import torch.nn as nn
from d_vae import DallE_VAE
from torch.utils.data import DataLoader, Dataset
from dall_e.utils import map_pixels    
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.datasets import ImageNet
import os
from PIL import Image

from vision_transformer import VisionTransformer

class Dat(Dataset):
    def __init__(self, root, t):
        super().__init__()
        self.data_dir = root
        self.transform = t
        self.images = []
        self.labels = []
        self.class_mapping = {class_name: label for label, class_name in enumerate(os.listdir('tiny-imagenet-200/train'))}
        class_folders = os.listdir(root)
        for class_folder in class_folders:
            if not class_folder.endswith(".txt"):
                if class_folder != "images":
                    class_folder_path = os.path.join(root, class_folder, "images")
                else:
                    class_folder_path = os.path.join(root, class_folder)
            for image_name in os.listdir(class_folder_path):
                if image_name.endswith(".JPEG"):
                    self.images.append(os.path.join(class_folder_path, image_name))
                    self.labels.append(class_folder)

        self.n_images = len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img = Image.open(img_name).convert('RGB')
        img_full_size = self.transform(img)
        return img_full_size, self.class_mapping[self.labels[idx]]
    
    def __len__(self):
        return self.n_images
    
#write a convolutional neural network (simple)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.cv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.linear = nn.Linear(16384, 200)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = F.relu(self.cv1(x))
        x = F.relu(self.cv2(x))
        x = x.flatten(1)
        x = self.linear(x)
        return x

if __name__ == "__main__":
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
    train_dataset = Dat(root='tiny-imagenet-200/train', t = base_transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True, num_workers=4)

    class_mapping = {class_name: label for label, class_name in enumerate(os.listdir('tiny-imagenet-200/train'))}
    print(class_mapping['n01443537'])

    model = CNN().to("cuda")
    learning_rate = 1.5e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    first = True
    for epoch in range(5):
        for i, (img_full_size, label) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(img_full_size.to("cuda"))
            if first:
                print(output.shape)
                print(label.shape)
                first = False
            
            loss = criterion(output, label.to("cuda"))
            loss.backward()
            optimizer.step()
            print(loss)

    if False:
        dVae = DallE_VAE(vt_image_size).to("cuda")
        dVae.load_model("dall_e/models/",  "cuda")
        image_size =64
        model = VisionTransformer(img_size = image_size, return_all_tokens=True, patch_size=8, embed_dim=192, depth = 8).to("cuda")

        for i, (img_full_size, img_half_size, _) in enumerate(train_loader):
            print(img_full_size.shape)
            print(img_half_size.shape)
            plt.imshow(img_half_size[0].permute(1, 2, 0).cpu().detach().numpy())
            plt.show()
            #codebook_idx = dVae.get_codebook_indices(img_half_size.to("cuda"))
            #output = dVae.decode(codebook_idx)
            output, _ = model(img_full_size.to("cuda").float())
            ss = output
            print(output.shape)
            output = dVae.decode(torch.argmax(output, 2))
            print(output.shape)
            plt.imshow(output[0, 0:3, :].permute(1, 2, 0).cpu().detach().numpy())
            plt.show()
            plt.imshow(output[0, 3:6, :].permute(1, 2, 0).cpu().detach().numpy())
            plt.show()
            print(ss.shape)
            print(ss.max(-1))
            print(ss.max(-1)[1])
            break
