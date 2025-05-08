from models.vision_transformer import ViT
from models.language_model import LanguageModel
from models.modality_projector import ModalityProjector

import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionLanguageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.vision_encoder = ViT(cfg)
        self.decoder = LanguageModel(cfg)
        self.MP = ModalityProjector(cfg)

    def forward(self, input_ids, image, attention_mask=None, targets=None):
        image_embd = self.vision_encoder(image)
        image_embd = self.MP(image_embd)

        token_embd = self.decoder.token_embedding(input_ids)

        combined_embd = torch.cat((image_embd, token_embd), dim=1) # Concatenate image embeddings to token embeddings

        # Adjust attention mask to account for image tokens
        if attention_mask is not None:
            # Create mask of 1s for image tokens (all image tokens should be attended to)
            batch_size = image_embd.size(0)
            img_seq_len = image_embd.size(1)
            image_attention_mask = torch.ones((batch_size, img_seq_len), device=attention_mask.device, dtype=attention_mask.dtype)

            # Combine image and token attention masks
            attention_mask = torch.cat((image_attention_mask, attention_mask), dim=1)

        # Pass combined_embd to the decoder. 
        # If lm_use_tokens is False (VLM default), decoder's forward returns normalized hidden states.
        # If lm_use_tokens is True, decoder's forward returns logits.
        output_from_decoder = self.decoder(combined_embd, attention_mask=attention_mask, use_cache=False) 

        loss = None
        if targets is not None:
            logits = output_from_decoder
            if not self.decoder.lm_use_tokens: # Apply head if decoder outputted hidden states
                logits = self.decoder.head(logits)
            
            logits = logits[:, image_embd.size(1):, :] # Use only token part for loss
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        
        return output_from_decoder, loss

    @torch.no_grad()
    def generate(self, input_ids, image, attention_mask=None, max_new_tokens=5):
        # Process image and initial text tokens
        image_embd = self.vision_encoder(image)
        image_embd = self.MP(image_embd)
        token_embd = self.decoder.token_embedding(input_ids)
        initial_embeddings = torch.cat((image_embd, token_embd), dim=1)

        batch_size = initial_embeddings.size(0)
        prompt_len = initial_embeddings.size(1)
        
        # Prepare attention mask for the initial prompt processing
        prompt_attention_mask = None
        if attention_mask is not None:
            image_attention_mask = torch.ones((batch_size, image_embd.size(1)), device=attention_mask.device, dtype=attention_mask.dtype)
            prompt_attention_mask = torch.cat((image_attention_mask, attention_mask), dim=1)

        # KV cache initialization
        kv_caches = [None] * len(self.decoder.blocks)
        prompt_pos_ids = torch.arange(prompt_len, device=initial_embeddings.device).unsqueeze(0).expand(batch_size, -1)
        prompt_cos, prompt_sin = self.decoder.rotary_embd(prompt_pos_ids)
        
        hidden_states = initial_embeddings
        for i, block_module in enumerate(self.decoder.blocks):
            hidden_states, kv_caches[i] = block_module(hidden_states, prompt_cos, prompt_sin, prompt_attention_mask, None, True)
        prompt_final_hidden_state_normed = self.decoder.norm(hidden_states[:, -1, :]) # H_prompt_last_normed

        # Autoregressive generation loop
        generated_tokens_ids = torch.zeros((batch_size, max_new_tokens), device=input_ids.device, dtype=input_ids.dtype)
        next_token_embeddings = None # Stores E_k for generating token k+1

        for k in range(max_new_tokens): # k: index of the new token being generated
            if k == 0:
                # First token: logits from H_prompt_last_normed
                current_logits = self.decoder.head(prompt_final_hidden_state_normed)
                # cfg.lm_use_tokens is False for VLM, so head is applied here.
            else:
                # Subsequent tokens: process E_(k-1) (embedding of previously generated token)
                current_input_for_blocks = next_token_embeddings # E_(k-1) [B, 1, C]
                
                # RoPE for E_(k-1) at its absolute position (prompt_len + k - 1)
                pos_ids_for_input_token = torch.tensor([[prompt_len + k - 1]], device=initial_embeddings.device).expand(batch_size, -1)
                cos_current, sin_current = self.decoder.rotary_embd(pos_ids_for_input_token)
                
                hidden_output_current_step = current_input_for_blocks
                for i_block, block_module in enumerate(self.decoder.blocks):
                    hidden_output_current_step, kv_caches[i_block] = block_module(hidden_output_current_step, cos_current, sin_current, None, kv_caches[i_block], True)
                
                current_new_token_hidden_normed = self.decoder.norm(hidden_output_current_step) # H_k_normed [B, 1, C]
                current_logits = self.decoder.head(current_new_token_hidden_normed.squeeze(1)) # Squeeze to [B, C] for head

            # Sample token_k
            probs = F.softmax(current_logits, dim=-1)
            sampled_token_k_ids = torch.multinomial(probs, num_samples=1) # [B, 1]
            generated_tokens_ids[:, k] = sampled_token_k_ids.squeeze(-1)
            
            # Prepare E_k for the next iteration
            next_token_embeddings = self.decoder.token_embedding(sampled_token_k_ids) # E_k [B, 1, C]
            
        return generated_tokens_ids
        
    def load_checkpoint(self, path):
        print(f"Loading weights from full VLM checkpoint: {path}")
        checkpoint = torch.load(path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))        
        self.load_state_dict(checkpoint)

    @classmethod
    def from_pretrained(cls, cfg):
        model = cls(cfg)
        model.vision_encoder = ViT.from_pretrained(cfg)
        model.decoder = LanguageModel.from_pretrained(cfg)

        return model