import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import datasets
import argparse

class MultiCropAugment:
    def __init__(self, global_size=448, local_size=160, local_crops=0):
        self.local_crops = local_crops

        color = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)

        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(global_size, scale=(0.5, 1.0)),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomApply([color], p=0.8),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
        ])

        self.local_t = transforms.Compose([
            transforms.RandomResizedCrop(local_size, scale=(0.2, 0.6), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomApply([color], p=0.8),
            transforms.RandomGrayscale(p=0.05),
            transforms.GaussianBlur(kernel_size=15, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
        ])

        def __call__(self, img):
            crops = [self.global_t(img), self.global_t(img)]
            for _ in range(self.local_crops):
                crops.append(self.local_t(img))
            return crops

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="ImageFolder directory (train split).", default="../dataset/images")
    ap.add_argument("--vit", type=str, default="vit_base_patch16_224", help="timm ViT name")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)  # effective batch is per-crop, so keep sane
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--global_size", type=int, default=224)
    ap.add_argument("--local_size", type=int, default=96)
    ap.add_argument("--local_crops", type=int, default=6)
    ap.add_argument("--out_dim", type=int, default=65536)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--save_path", type=str, default="vit_dino_student.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    augment = MultiCropAugment(args.global_size, args.local_size, args.local_crops)

    ds = datasets.ImageFolder(root=args.data_dir, transform=augment)

    def collate_multicrop(batch):
        all_crops = []
        for crops, _ in batch:
            all_crops.extend(crops)
        return all_crops

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_multicrop,
    )