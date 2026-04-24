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

from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training, PeftModel

from ChartLLM import ChartLLM, Connector

if __name__ == "__main__":

    #  Get directory of repository
    MAIN_DIR = Path(__file__).resolve().parent.parent

    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--vit_name", type=str, help="Name of ViT architecture to use (huggingface).", default="openai/clip-vit-large-patch14")
    ap.add_argument("--vit_weights", type=str, help="Path to ViT weights file.", default="../models/vit_clip_openai_clip-vit-large-patch14.pth")
    ap.add_argument("--llm_name", type=str, help="Name of LLM architecture to use (huggingface).", default="Qwen/Qwen3-32B")
    ap.add_argument("--quantization", action="store_true", help="Whether to use 4-bit quantization.")
    ap.add_argument("--lora", action="store_true", help="Whether to use LoRA.")
    ap.add_argument("--lora_weights", type=str, help="Path to LoRA weights file (if --lora is enabled).", default="../models/llm_openai_clip-vit-large-patch14_Qwen_Qwen3-32B_lora")
    ap.add_argument("--connector_weights", type=str, help="Path to connector weights file.", default="../models/connector_openai_clip-vit-large-patch14_Qwen_Qwen3-32B.pth")
    args = ap.parse_args()

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

    vit_model.to(device)

    # Inizialize LLM with or without quantization
    if args.quantization:
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
    else:
        llm_model = AutoModelForCausalLM.from_pretrained(args.llm_name, torch_dtype=torch.float16)
        llm_model.to(device)  

    # Add lora weights if LoRA enabled
    if args.lora:
        llm_model = PeftModel.from_pretrained(llm_model, args.lora_weights)
    
    # Initialize connector
    connector = Connector(vit_dim=vit_model.config.hidden_size, llm_dim=llm_model.config.hidden_size)
    connector_state_dict = torch.load(args.connector_weights, map_location="cpu")
    connector.load_state_dict(connector_state_dict)
    connector.to(device, dtype=torch.bfloat16¸w)

    chart_llm = ChartLLM(vit_model=vit_model, llm_model=llm_model, connector=connector)
    chart_llm.eval()



    
