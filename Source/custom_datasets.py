from typing import Callable
import os
from PIL import Image
from torch.utils.data import Dataset

class Food101N(Dataset):
    """Food-101N Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``food-101n/images`` and ``food-101n/meta`` exist.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version.
    """

    def __init__(self, root: str, transform: Callable = None) -> None:
        self.root = os.path.join(root, 'food-101n')
        self.transform = transform

        # Read the class list from meta/classes.txt
        class_file = os.path.join(self.root, 'meta', 'classes.txt')
        with open(class_file, 'r') as f:
            self.classes = f.read().splitlines()

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []
        images_root = os.path.join(self.root, 'images')
        # For each class directory in images/
        for class_name in sorted(os.listdir(images_root)):
            class_dir = os.path.join(images_root, class_name)
            if not os.path.isdir(class_dir):
                continue
            class_idx = self.class_to_idx[class_name]
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.jpg'):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))

    
    def __len__(self) -> int:
        return len(self.samples)
    

    def __getitem__(self, idx: int):
        img_path, target = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target



class WebVision(Dataset):
    def __init__(self, root: str, split: str, transform: Callable = None) -> None:
        self.root = os.path.join(root, 'webvision')
        self.transform = transform

        class_file = os.path.join(self.root, 'imagenet1000_classes.txt')
        with open(class_file, 'r') as f:
            self.classes = f.read().splitlines()

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []
        if split not in ['train', 'val']:
            raise ValueError(f"split must be either 'train' or 'val', got {split}")
        elif split == 'train':
            with open(os.path.join(self.root, 'train_filelist_google.txt'), 'r') as f:
                lines = f.read().splitlines()
            for line in lines:
                img_path, class_idx = line.split()
                img_path = os.path.join(self.root, img_path)
                self.samples.append((img_path, int(class_idx)))

            with open(os.path.join(self.root, 'train_filelist_flickr.txt'), 'r') as f:
                lines = f.read().splitlines()
            for line in lines:
                img_path, class_idx = line.split()
                img_path = os.path.join(self.root, img_path)
                self.samples.append((img_path, int(class_idx)))
        else:
            val_label_file = os.path.join(self.root, 'val_filelist.txt')
            image_folder = os.path.join(self.root, 'val_images_256')
            with open(val_label_file, 'r') as f:
                lines = f.read().splitlines()
            for line in lines:
                img_path, class_idx = line.split()
                img_path = os.path.join(image_folder, img_path)
                self.samples.append((img_path, int(class_idx)))

        print(f"Loaded {len(self.samples)} samples from {split} split")

    def __len__(self) -> int:
        return len(self.samples)
    

    def __getitem__(self, idx: int):
        img_path, target = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target