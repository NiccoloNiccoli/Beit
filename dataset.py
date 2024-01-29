import os
import torch.utils.data as data
from PIL import Image
import json

from masking_generator import MaskingGenerator

class ImageNetDataset (data.Dataset):

    def __init__(self, root, split, base_transform = None, patch_transform = None, vae_transform = None):
        self.samples = []
        self.targets = []
        self.base_transform = base_transform
        self.patch_transform = patch_transform
        self.vae_transform = vae_transform
        self.syn_to_class = {}
        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
                    json_file = json.load(f)
                    for class_id, v in json_file.items():
                        self.syn_to_class[v[0]] = int(class_id)
        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
                    self.val_to_syn = json.load(f)
        samples_dir = os.path.join(root, "Data\CLS-LOC", split)
        for entry in os.listdir(samples_dir):
            if split == "train":
                syn_id = entry
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)
            elif split == "val":
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)

        #generate mask
        self.masekd_position_generator = MaskingGenerator(input_size = 14, n_masking_patches=75, max_masking_factor=0.4)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = Image.open(self.samples[idx]).convert("RGB")
        if self.base_transform is not None and self.patch_transform is not None and self.vae_transform is not None:
             x_patches, x_vae = self.base_transform(x)
        return self.patch_transform(x_patches), self.vae_transform(x_vae), self.masekd_position_generator(), self.targets[idx]
               