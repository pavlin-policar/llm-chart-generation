import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import argparse
import os
import math
from pathlib import Path


from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor
from transformers import CLIPProcessor, CLIPVisionModel, CLIPModel
from transformers import BitsAndBytesConfig
import tqdm
import json
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

from ChartLLM import ChartLLM, Connector

# VLMDataset (make it so it returns different prompts for different samples)
class VLMDataset(Dataset):
    def __init__(self, data_path, dir_path=None, tokenizer=None, fixed_prompt="Describe the graph in detail."):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.dir_path = dir_path
        self.samples = []

        # (Change this so random prompt is returned for each sample)
        self.fixed_prompt_ids = tokenizer(
            fixed_prompt,
            return_tensors="pt",
            truncation=True
        ).input_ids.squeeze(0)

        with open(os.path.join(self.data_path, "metadata.jsonl"), "r") as f:
            for line in f:
                sample = json.loads(line)
                self.samples.append((os.path.join(self.dir_path, sample["images"][0]["path"]), sample["graph"]["full_description"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, description = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        target_ids = self.tokenizer(
            description,
            return_tensors="pt",
            truncation=True,
            max_length=2048, 
        ).input_ids.squeeze(0)

        return image, self.fixed_prompt_ids, target_ids

# Collate function
def make_collate(tokenizer, processor):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    def collate(batch):
        images, prompt_ids, target_ids = zip(*batch)

        vision_inputs = processor(
            images=list(images),
            return_tensors="pt"
        )

        pixel_values = vision_inputs["pixel_values"]

        prompt_ids = pad_sequence(prompt_ids, batch_first=True, padding_value=pad_id)
        target_ids = pad_sequence(target_ids, batch_first=True, padding_value=pad_id)

        return pixel_values, prompt_ids, target_ids

    return collate


def train(model: ModalLLM, dataloader, device, total_epochs=100, out_path="connector_weights.pth", lora_out_path=None, lr=3e-5, weight_decay=0.01):
    model.train()

    total_steps = total_epochs * len(dataloader)
    step = 0

    train_params = list(model.connector.parameters()) + [
        p for p in model.llm.parameters() if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(train_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    print("Starting training...")

    avg_loss = 0.0

    for epoch in range(total_epochs):
        for i, b in enumerate(dataloader):

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                images, prompt_ids, target_ids = b
                images = images.to(device)
                prompt_ids = prompt_ids.to(device)
                target_ids = target_ids.to(device)

                outputs = model(images, prompt_ids, target_ids)

            loss = outputs.loss
            avg_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.connector.parameters(),
                max_norm=1.0
            )

            optimizer.step()
            scheduler.step()

            if step % 10 == 0 and step > 0:
                print(f"Epoch {epoch}, Step: {step}/{total_steps}, Loss: {avg_loss / (10):.4f}")
                avg_loss = 0.0

            step += 1

    torch.save(model.connector.state_dict(), out_path)

    # Save LoRA weights if using LoRA
    if lora_out_path is not None:
        model.llm.save_pretrained(lora_out_path)

if __name__ == "__main__":

    # Get directory of repository
    MAIN_DIR = Path(__file__).resolve().parent.parent

    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, help="ImageFolder directory (train split).", default="../dataset")
    ap.add_argument("--vit_name", type=str, help="Name of ViT architecture to use (huggingface).", default="openai/clip-vit-large-patch14")
    ap.add_argument("--vit_weights", type=str, help="Path to ViT weights file.", default="../models/vit_clip_openai_clip-vit-large-patch14.pth")
    ap.add_argument("--llm_name", type=str, help="Name of LLM architecture to use (huggingface).", default="Qwen/Qwen3-32B")
    ap.add_argument("--batch_size", type=int, help="Batch size (number of images, not number of crops).", default=2)
    ap.add_argument("--epochs", type=int, help="Number of epochs to train for.", default=1)
    ap.add_argument("--lr", type=float, help="Learning rate for student.", default=1e-4)
    ap.add_argument("--weight_decay", type=float, help="Weight decay for student.", default=0.01)
    ap.add_argument("--out", type=str, help="Path to save the trained model.", default="")
    ap.add_argument("--lora", action="store_true", help="Whether to use LoRA for training the connector.")
    ap.add_argument("--lora_rank", type=int, help="LoRA rank (if --lora is set).", default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_out", type=str, help="Path to save the LoRA weights (if --lora is set).", default="")
    ap.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj", # Possible values q_proj,k_proj,v_proj,o_proj
        help="Comma-separated module names to LoRA-ize."
    )
    
    args = ap.parse_args()

    # Process LoRA target modules and check if valid
    args.lora_target_modules = args.lora_target_modules.strip().split(",")
    for i in args.lora_target_modules:
        if i not in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            raise ValueError(f"Invalid LoRA target module: {i}. Must be one of q_proj, k_proj, v_proj, o_proj.")
    
    # Set default output path if not provided
    if args.out == "":
        args.out = os.path.join(MAIN_DIR, "models", f"connector_{args.vit_name.replace('/', '_')}_{args.llm_name.replace('/', '_')}.pth")
        
    if args.lora_out == "":
        args.lora_out = os.path.join(MAIN_DIR, "models", f"llm_{args.vit_name.replace('/', '_')}_{args.llm_name.replace('/', '_')}_lora")

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create LLM Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llm_name)

    # Read ViT weights and initialize ViT model (either CLIP or DINOv2 depending on the architecture)
    if args.vit_weights != None:
        state_dict = torch.load(args.vit_weights, map_location="cpu")
        
    if args.vit_name == "facebook/dinov2-base":
        processor = AutoImageProcessor.from_pretrained(args.vit_name)
        vit_model = AutoModel.from_pretrained(args.vit_name, torch_dtype=torch.float16)
        vit_model.load_state_dict(state_dict, strict=False)
    else:
        processor = CLIPProcessor.from_pretrained(args.vit_name)
        vit_model = CLIPVisionModel.from_pretrained(args.vit_name, torch_dtype=torch.float16)
        missing, unexpected = vit_model.load_state_dict(state_dict, strict=False)
        
        assert len(missing) == 0, f"Missing keys in ViT state dict: {missing}, unexpected keys: {unexpected}"

    # Freeze ViT
    vit_model = vit_model.to(device)
    vit_model.eval()
    for p in vit_model.parameters():
        p.requires_grad = False

    # Inizialize LLM and freeze it
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,  # A100 loves bf16
    )

    llm_model = AutoModelForCausalLM.from_pretrained(
        args.llm_name,
        quantization_config=bnb_config,
        device_map={"": 0},   # single A100
    )

    if args.lora:
        print("Using LoRA for training the connector...")

        # Prepare 4-bit model for LoRA
        llm_model = prepare_model_for_kbit_training(llm_model)

        # Configure and initialize peft model
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            bias="none",
        )
        llm_model = get_peft_model(llm_model, lora_cfg)
        llm_model.print_trainable_parameters()

    else:
        print("Training connector without LoRA...")

        # Freeze full LLM since we're not using LoRA
        for p in llm_model.parameters():
            p.requires_grad = False
            
    llm_type = next(llm_model.parameters()).dtype

    # Disable caching and enable gradient checkpointing for LLM to save VRAM (idk if this is needed?)
    llm_model.config.use_cache = False
    llm_model.gradient_checkpointing_enable()

    # Initialize connector
    connector = Connector(vit_dim=vit_model.config.hidden_size, llm_dim=llm_model.config.hidden_size)
    connector = connector.to(device, dtype=llm_type)

    # Check types and devices
    print("LLM dtype:", llm_type)
    print("LLM device:", next(llm_model.parameters()).device)

    print("Connector dtype:", next(connector.parameters()).dtype)
    print("Connector device:", next(connector.parameters()).device)

    print("VIT dtype:", next(vit_model.parameters()).dtype)
    print("VIT device:", next(vit_model.parameters()).device)

    # Initialize full model
    chart_llm = ChartLLM(vit_model=vit_model, llm_model=llm_model, connector=connector)

    # Create dataset and dataloader
    ds = VLMDataset(data_path=args.data_dir, dir_path=MAIN_DIR, tokenizer=tokenizer)

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,         
        pin_memory=True,
        persistent_workers=True,
        collate_fn=make_collate(tokenizer, processor)
    )

    train(chart_llm, dl, device, total_epochs=args.epochs, out_path=args.out, lr=args.lr, weight_decay=args.weight_decay, lora_out_path=args.lora_out if args.lora else None)
