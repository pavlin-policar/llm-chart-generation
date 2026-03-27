import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import argparse
import os
import math
from pathlib import Path

from transformers import AutoModel
import tqdm

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

class MultiCropAugment:
    def __init__(self, global_size=448, local_size=160, local_crops=0):
        self.local_crops = local_crops

        color = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)

        self.global_t = transforms.Compose([
            transforms.RandomResizedCrop(global_size, scale=(0.5, 1.0)),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomApply([color], p=0.8),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225))
        ])

        self.local_t = transforms.Compose([
            transforms.RandomResizedCrop(local_size, scale=(0.2, 0.6), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomApply([color], p=0.8),
            transforms.RandomGrayscale(p=0.05),
            transforms.GaussianBlur(kernel_size=15, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225))
        ])

    def __call__(self, img):
        crops = [self.global_t(img), self.global_t(img)]
        for _ in range(self.local_crops):
            crops.append(self.local_t(img))
        return crops


def collate_multicrop(batch):
    n_crops = len(batch[0][0])
    out = [[] for _ in range(n_crops)]
    for crops, _ in batch:
        for i, c in enumerate(crops):
            out[i].append(c)
    return [torch.stack(v, 0) for v in out]  # list of tensors, each [B,3,H,W]

class DINOv2Head(nn.Module):
    def __init__(self, in_dim, out_dim=65536, hidden_dim=2048, bottleneck_dim=256, nlayers=3):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(nlayers - 1):
            layers += [nn.Linear(dim, hidden_dim), nn.GELU()]
            dim = hidden_dim
        layers += [nn.Linear(dim, bottleneck_dim)]
        self.mlp = nn.Sequential(*layers)

        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1.0)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        x = self.last_layer(x)
        return x
        

class StudentTeacher(nn.Module):
    def __init__(self, vit_name: str, out_dim: int):
        super().__init__()

        self.student_backbone =  AutoModel.from_pretrained(vit_name)
        feat_dim = self.student_backbone.config.hidden_size
        self.student_head = DINOv2Head(feat_dim, out_dim=out_dim)

        self.teacher_backbone = AutoModel.from_pretrained(vit_name)
        self.teacher_head = DINOv2Head(feat_dim, out_dim=out_dim)

        self._init_teacher()

        for p in self.teacher_backbone.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _init_teacher(self):
        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict(), strict=True)
        self.teacher_head.load_state_dict(self.student_head.state_dict(), strict=True)

    def student(self, x):
        out = self.student_backbone(pixel_values=x, interpolate_pos_encoding=True)
        feats = out.last_hidden_state[:, 0]
        return self.student_head(feats)

    @torch.no_grad()
    def teacher(self, x):
        out = self.teacher_backbone(pixel_values=x, interpolate_pos_encoding=True)
        feats = out.last_hidden_state[:, 0]
        return self.teacher_head(feats)

    @torch.no_grad()
    def ema_update_teacher(self, m: float):
        for ps, pt in zip(self.student_backbone.parameters(), self.teacher_backbone.parameters()):
            pt.data.mul_(m).add_(ps.data, alpha=(1.0 - m))
        for ps, pt in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            pt.data.mul_(m).add_(ps.data, alpha=(1.0 - m))

class DINOv2Loss(nn.Module):
    def __init__(self, out_dim: int = 65536, student_temp: float = 0.1, teacher_temp_final: float = 0.04, teacher_temp_warmup: float = 0.1, warmup_epochs: int = 8, center_momentum: float = 0.9):
        super().__init__()

        self.out_dim = out_dim
        self.student_temp = student_temp
        self.teacher_temp_final = teacher_temp_final
        self.teacher_temp_warmup = teacher_temp_warmup
        self.warmup_epochs = warmup_epochs

        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def teacher_temp(self, epoch):
        if epoch < self.warmup_epochs:
            return self.teacher_temp_warmup + epoch / self.warmup_epochs * (self.teacher_temp_final - self.teacher_temp_warmup)
        else:
            return self.teacher_temp_final

    @torch.no_grad()
    def update_center(self, teacher_logits):
        batch_center = torch.mean(teacher_logits, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def forward(self, student_out, teacher_out, epoch):
        loss = 0
        n_loss_terms = 0
        ttemp = self.teacher_temp(epoch) # Get teacher temparature for epoch

        # Compute teacher probabilities and update center
        with torch.no_grad():
            teacher_probs = []
            for t in teacher_out:
                t = (t - self.center) / ttemp
                teacher_probs.append(F.softmax(t, dim=-1))
            self.update_center(torch.cat(teacher_out, dim=0))

        # Compute student log probabilities
        student_logp = []
        for s in student_out:
            s = s / self.student_temp
            student_logp.append(F.log_softmax(s, dim=-1))

        # Compute CE loss
        for i, t in enumerate(teacher_probs):
            for j, s in enumerate(student_logp):
                if i == j:
                    continue
                loss += - (t * s).sum(dim=-1).mean()
                n_loss_terms += 1

        return loss / max(1, n_loss_terms)

def cosine_schedule(base, final, step, total_steps):
    if total_steps <= 1:
        return final
    t = step / (total_steps - 1)
    return final + 0.5 * (base - final) * (1 + math.cos(math.pi * t))

if __name__ == "__main__":
    MAIN_DIR = Path(__file__).resolve().parent.parent

    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)


    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, help="ImageFolder directory (train split).", default="../dataset")
    ap.add_argument("--vit_name", type=str, help="Name of ViT architecture to use (huggingface).", default="facebook/dinov2-base")
    ap.add_argument("--global_size", type=int, help="Size of global crops.", default=518)
    ap.add_argument("--local_size", type=int, help="Size of local crops.", default=196)
    ap.add_argument("--local_crops", type=int, help="Number of local crops to generate.", default=8)
    ap.add_argument("--batch_size", type=int, help="Batch size (number of images, not number of crops).", default=32)
    ap.add_argument("--lr", type=float, help="Learning rate for student.", default=3e-5)
    ap.add_argument("--weight_decay", type=float, help="Weight decay for student.", default=0.04)
    ap.add_argument("--epochs", type=int, help="Number of epochs to train for.", default=100)
    ap.add_argument("--out", type=str, help="Path to save the trained model.", default="")
    args = ap.parse_args()

    if args.out == "":
        args.out = os.path.join(MAIN_DIR, "models", f"vit_dinov2_{args.vit_name.replace('/', '_')}.pth")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(device)

    augment = MultiCropAugment(args.global_size, args.local_size, args.local_crops)

    ds = datasets.ImageFolder(root=args.data_dir, transform=augment)

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_multicrop,
        num_workers=16,         
        pin_memory=True,
        persistent_workers=True,
    )

    print("Build Dataloader done.")

    model = StudentTeacher(args.vit_name, out_dim=65536).to(device)
    criterion = DINOv2Loss().to(device)

    optim = torch.optim.AdamW(
        list(model.student_backbone.parameters()) + list(model.student_head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scaler = torch.amp.GradScaler(device=device)

    total_steps = args.epochs * len(dl)
    step = 0

    print("Create model and optimizer done.")

    model.train()
    model.teacher_backbone.eval()
    model.teacher_head.eval()
    for epoch in range(args.epochs):
        print(f"Starting epoch {epoch+1}/{args.epochs}...")
        for crops in dl:

            # crops is typically a list of tensors: [2 + n_local] items, each [B,3,H,W]
            crop_list = [c.to(device, non_blocking=True) for c in crops]
            global_crops = crop_list[:2]
            local_crops  = crop_list[2:]

            optim.zero_grad(set_to_none=True)

            m = cosine_schedule(0.99, 0.9999, step, total_steps)

            autocast_enabled = (isinstance(device, str) and device == "cuda") or (
                hasattr(device, "type") and device.type == "cuda"
            )

            with torch.cuda.amp.autocast(enabled=autocast_enabled, dtype=torch.float16):
                # Global crops
                g_batch = torch.cat(global_crops, dim=0)
                g_out = model.student(g_batch)  
                B = global_crops[0].shape[0]
                student_global = list(g_out.split(B, dim=0))  # len=2, each [B,...]

                # Local crops
                if len(local_crops) > 0:
                    l_batch = torch.cat(local_crops, dim=0)
                    l_out = model.student(l_batch)
                    student_local = list(l_out.split(B, dim=0))
                else:
                    student_local = []

                student_out = student_global + student_local

                with torch.no_grad():
                    t_out = model.teacher(g_batch)
                    teacher_out = list(t_out.split(B, dim=0))

                loss = criterion(student_out, teacher_out, epoch)

            scaler.scale(loss).backward()

            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(
                list(model.student_backbone.parameters()) + list(model.student_head.parameters()),
                max_norm=1.0
            )
            scaler.step(optim)
            scaler.update()

            # EMA update teacher
            with torch.no_grad():
                model.ema_update_teacher(m)

            # LR cosine decay
            lr_now = cosine_schedule(args.lr, args.lr * 0.1, step, total_steps)
            for pg in optim.param_groups:
                pg["lr"] = lr_now

            if step >= 0:
                print(f"epoch={epoch} step={step}/{total_steps} loss={loss.item():.4f} lr={lr_now:.2e} ema_m={m:.5f}")

            if step % 50 == 0:
                with torch.no_grad():
                    s_std = torch.cat(student_out, 0).std(dim=0).mean().item()
                    t_std = torch.cat(teacher_out, 0).std(dim=0).mean().item()
                    print(f"feat_std student={s_std:.4f} teacher={t_std:.4f}")

            step += 1


    # Save only the student backbone (your encoder)
    torch.save(model.student_backbone.state_dict(), args.save_path)
    print(f"Saved student ViT encoder to: {args.save_path}")

        



