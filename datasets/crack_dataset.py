import os.path
import cv2
import random
import numpy as np
import torch
from PIL import Image
from .base_dataset import BaseDataset
import torchvision.transforms as transforms
from .image_folder import make_dataset
from .utils import MaskToTensor

class CrackDataset(BaseDataset):
    def __init__(self, args):
        BaseDataset.__init__(self, args)
        self.modals = args.modals
        self.miss_rates = args.miss_rates
        self.modal_num = len(self.modals)

        if len(self.miss_rates) != self.modal_num:
            raise ValueError("The length of miss_rates must be the same as that of modals.")

        self.modal_img_paths = []
        for modal in self.modals:
            modal_dir = f"{args.phase}_img_{modal}"
            paths = make_dataset(os.path.join(args.dataset_path, modal_dir))
            self.modal_img_paths.append(paths)

        self.lab_dir = os.path.join(args.dataset_path, f"{args.phase}_lab")

        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.lab_transform = MaskToTensor()

    def __getitem__(self, index):
        returned_data = {}
        image_paths = []
        modal_mask = []
        w, h = self.args.load_width, self.args.load_height

        for i in range(self.modal_num):
            img_path = self.modal_img_paths[i][index]
            image_paths.append(img_path)

            miss_rate = self.miss_rates[i]
            if random.random() < miss_rate:

                noise = torch.randn(3, h, w)
                returned_data[self.modals[i]] = noise
                modal_mask.append(0.0)
            else:
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
                returned_data[self.modals[i]] = self.img_transforms(Image.fromarray(img))
                modal_mask.append(1.0)

        returned_data['modal_mask'] = torch.tensor(modal_mask, dtype=torch.float32)

        base_name = os.path.basename(image_paths[0]).split('.')[0]
        lab_path = os.path.join(self.lab_dir, f"{base_name}.png")
        lab = cv2.imread(lab_path, cv2.IMREAD_UNCHANGED)

        if len(lab.shape) == 3:
            lab = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)
        lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_NEAREST)
        _, lab = cv2.threshold(lab, 127, 1, cv2.THRESH_BINARY)
        returned_data['label'] = self.lab_transform(lab).unsqueeze(0)

        returned_data['image_path'] = image_paths[0]
        returned_data['label_path'] = lab_path

        return returned_data

    def __len__(self):
        return len(self.modal_img_paths[0])
