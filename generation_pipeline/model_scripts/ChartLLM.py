import torch
import torch.nn as nn

class ChartLLM(nn.Module):
    def __init__(self, vit_model, llm_model, connector):
        super(ChartLLM, self).__init__()
        self.vit = vit_model
        self.llm = llm_model
        self.connector = connector
        self.pad_id = self.llm.config.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.llm.config.eos_token_id

    def forward(self, images, prompt_ids, target_ids=None):
        llm_dtype = next(self.llm.parameters()).dtype
        llm_device = next(self.llm.parameters()).device

        # Get text emebeddings for prompt and target
        txt_prompt = self.llm.get_input_embeddings()(prompt_ids)
        txt_target = None
        if target_ids is not None:
            txt_target = self.llm.get_input_embeddings()(target_ids)

        # Get ViT outputs and pass through connector
        vit_outputs = self.vit(images)
        vit_outputs = vit_outputs.last_hidden_state.to(dtype=llm_dtype)
        img_embeds = self.connector(vit_outputs)

        inputs_embeds = torch.cat([img_embeds, txt_prompt, txt_target], dim=1)

        # Create labels, ignore image and prompt tokens (replace padding with -100 so it's are ignored in loss)
        B, Nv, _ = img_embeds.shape
        Tp = prompt_ids.shape[1]

        ignore = torch.full((B, Nv + Tp), -100, device=inputs_embeds.device)
        labels = torch.cat([ignore, target_ids], dim=1)
        labels[labels == self.pad_id] = -100

        # Get LLM outputs
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            labels=labels,
            return_dict=True,
        )

        return outputs

# Connector MLP between ViT and LLM
class Connector(nn.Module):
    def __init__(self, vit_dim, llm_dim):
        super(Connector, self).__init__()
        self.vit_dim = vit_dim
        self.llm_dim = llm_dim

        self.network = nn.Sequential(
            nn.Linear(vit_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim))

    def forward(self, vit_outputs):
        return self.network(vit_outputs)