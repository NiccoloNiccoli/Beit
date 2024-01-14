import torch
from d_vae import DallE_VAE
from vision_transformer import VisionTransformer
from dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from dall_e.utils import map_pixels

# Load the model and dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_transform = transforms.Compose([
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
test_dataset = Dataset(root='E:/Download/imagenette2-320/imagenette2-320/test', full_size_transform=base_transform, half_size_transform=visual_token_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=4)

model = VisionTransformer(return_all_tokens=False).to(device)
d_vae = DallE_VAE(112).to(device)
d_vae.load_model("dall_e/models/", device)

# Test the model on masked image modeling task
if False:
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        for i, (img_full_size, img_half_size, _) in enumerate(test_loader):
            outputs, mask = model(img_full_size.to(device))
            tokens = d_vae.get_codebook_indices(img_half_size.to(device)).flatten(1)
            tokens = tokens[mask]
            _, predicted = torch.max(outputs.data, 1)
            total += tokens.size(0)
            correct += (predicted == tokens).sum().item()
        accuracy = 100 * correct / total
        print(f"Accuracy on masked image modeling task: {accuracy}%")


