
import argparse, json, os, math, random
from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from transformers import CLIPModel, CLIPProcessor
from pathlib import Path


class PairDataset(Dataset):
    def __init__(self, data_path: str, dir_path: str = None, transform=None):
        self.samples = []
        self.data_path = data_path
        self.dir_path = dir_path
        self.transform = transform

        with open(os.path.join(self.data_path, "metadata.jsonl"), "r") as f:
            for line in f:
                sample = json.loads(line)
                self.samples.append((os.path.join(self.dir_path, sample["images"][0]["path"]), sample["graph"]["full_description"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img, text

def clip_loss(img_f, txt_f, logit_scale):
    img_f = F.normalize(img_f, dim=-1)
    txt_f = F.normalize(txt_f, dim=-1)
    logits = logit_scale * (img_f @ txt_f.t())
    targets = torch.arange(img_f.size(0), device=img_f.device)
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets)) / 2

if __name__ == "__main__":
    MAIN_DIR = Path(__file__).resolve().parent.parent

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, help="ImageFolder directory (train split).", default="../dataset")
    ap.add_argument("--model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--batch_size", type=int, help="Batch size (number of images, not number of crops).", default=32)
    ap.add_argument("--epochs", type=int, help="Number of epochs to train for.", default=20)
    ap.add_argument("--lr", type=float, help="Learning rate.", default=1e-6)
    ap.add_argument("--weight_decay", type=float, help="Weight decay.", default=0.01)
    ap.add_argument("--out", type=str, help="Path to save the trained model.", default="")

    args = ap.parse_args()

    if args.out == "":
        args.out = os.path.join(MAIN_DIR, "models", f"vit_clip_{args.model.replace('/', '_')}.pth")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained(args.model).to(device)
    processor = CLIPProcessor.from_pretrained(args.model)

    ds = PairDataset(args.data_dir, dir_path=MAIN_DIR)

    def collate(batch):
        images, texts = zip(*batch)
        inputs = processor(
            text=list(texts),
            images=list(images),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return inputs

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98), eps=1e-6)

    scaler = torch.amp.GradScaler(device=device)

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4, collate_fn=collate)
    step = 0
    all_steps = len(dl) * args.epochs

    print("Starting training...")

    for epoch in range(args.epochs):
        for inputs in dl:
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

            

            with torch.autocast(device_type=device, dtype=torch.float16):
                outputs = model(**inputs, return_dict=True)

                img_emb = outputs.image_embeds
                txt_emb = outputs.text_embeds

                model.logit_scale.data.clamp_(0, math.log(100))
                logit_scale = model.logit_scale.exp()

                loss = clip_loss(img_emb, txt_emb, logit_scale)

            optimizer.zero_grad(set_to_none=True)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()

            if step % 10 == 0 and step > 0:
                print(f"Epoch {epoch}, Step: {step}/{all_steps}, Loss: {loss.item():.4f}")

            step += 1

    print("Training complete. Saving model...")

    torch.save(model.state_dict(), args.out)
    print("saved:", args.out)