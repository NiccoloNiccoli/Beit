import os
import torch.utils.data as data
from PIL import Image
class Dataset(data.Dataset):
    """Custom definition of Dataset class in order to have a "full size" versions of the images for the ViT and a "half size" version for the Dall-E model.
    
    Parameters
    ----------
    root : str
        Path to the directory with the images.
    full_size_transform : torchvision.transforms.Compose
        Transformations to apply to the images for the ViT.
    half_size_transform : torchvision.transforms.Compose
        Transformations to apply to the images for the Dall-E model.
    """
    def __init__(self, root, full_size_transform, half_size_transform):
        super().__init__()
        self.data_dir = root
        self.full_size_transform = full_size_transform
        self.half_size_transform = half_size_transform
        self.images = []
        self.labels = []
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

    def __len__(self):
        return self.n_images
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img = Image.open(img_name).convert('RGB')
        img_full_size = self.full_size_transform(img)
        img_half_size = self.half_size_transform(img)
        return img_full_size, img_half_size, self.labels[idx]
    