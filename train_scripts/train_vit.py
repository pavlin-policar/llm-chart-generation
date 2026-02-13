import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class MultiView():
    def __init__(self, global_size=224, local_size=96, local_crops=0):
        self.local_crops = local_crops

        color = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)

        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(global_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([color], p=0.8),
            transforms.ToTensor(),
        ])