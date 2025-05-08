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
        # Encode image and text separately
        image_embd = self.vision_encoder(image)
        image_embd = self.MP(image_embd)
        token_embd = self.decoder.token_embedding(input_ids)

        # Concatenate image embeddings to token embeddings
        combined_embd = torch.cat((image_embd, token_embd), dim=1)
        
        # Adjust attention mask to account for image tokens
        if attention_mask is not None:
            batch_size = image_embd.size(0)
            img_seq_len = image_embd.size(1)
            image_attention_mask = torch.ones((batch_size, img_seq_len), device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat((image_attention_mask, attention_mask), dim=1)

        # Pass through decoder (returns hidden states if lm_use_tokens=False or logits if True)
        output_from_decoder = self.decoder(combined_embd, attention_mask=attention_mask, use_cache=False) 

        loss = None
        if targets is not None:
            # Apply head to get logits if decoder returned hidden states
            logits = output_from_decoder
            if not self.decoder.lm_use_tokens:
                logits = self.decoder.head(logits)
            
            # Use only the token part for loss computation
            logits = logits[:, image_embd.size(1):, :]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        
        return output_from_decoder, loss

    @torch.no_grad()
    def generate(self, input_ids, image, attention_mask=None, max_new_tokens=5):
        # Process image through vision encoder and projection
        image_embd = self.vision_encoder(image)
        image_embd = self.MP(image_embd)
        
        # Embed initial tokens
        token_embd = self.decoder.token_embedding(input_ids)
        
        # Concatenate image embeddings with token embeddings
        initial_embeddings = torch.cat((image_embd, token_embd), dim=1)

        batch_size = initial_embeddings.size(0)
        prompt_len = initial_embeddings.size(1)
        
        # Prepare attention mask for the initial prompt processing
        prompt_attention_mask = None
        if attention_mask is not None:
            image_attention_mask = torch.ones((batch_size, image_embd.size(1)), device=attention_mask.device, dtype=attention_mask.dtype)
            prompt_attention_mask = torch.cat((image_attention_mask, attention_mask), dim=1)
        
        # Initialize KV cache for each block in the decoder
        kv_caches = [None] * len(self.decoder.blocks)
        
        # Process initial prompt sequence to populate the KV cache
        prompt_pos_ids = torch.arange(prompt_len, device=initial_embeddings.device).unsqueeze(0).expand(batch_size, -1)
        prompt_cos, prompt_sin = self.decoder.rotary_embd(prompt_pos_ids)
        
        # Pass initial embeddings through all blocks to get hidden states
        current_hidden_states = initial_embeddings
        for i, block in enumerate(self.decoder.blocks):
            current_hidden_states, kv_caches[i] = block(current_hidden_states, prompt_cos, prompt_sin, prompt_attention_mask, None, True)
        
        # Get final hidden state of the prompt for the first token prediction
        prompt_last_hidden = self.decoder.norm(current_hidden_states[:, -1, :])  # [B, C]

        # Initialize tensor to store generated token IDs
        generated_tokens = torch.zeros((batch_size, max_new_tokens), device=input_ids.device, dtype=input_ids.dtype)
        
        # Store embeddings between generation steps
        next_token_embeddings = None

        for k in range(max_new_tokens):
            if k == 0:
                # For first new token, use last hidden state from prompt
                current_logits = self.decoder.head(prompt_last_hidden)
            else:
                # For subsequent tokens, use embedding of previous token
                current_input = next_token_embeddings  # [B, 1, C]
                
                # Get position for the current token
                pos_id = torch.tensor([[prompt_len + k - 1]], device=initial_embeddings.device).expand(batch_size, -1)
                cos, sin = self.decoder.rotary_embd(pos_id)
                
                # Process current token through blocks with KV cache
                hidden_state = current_input
                for i, block in enumerate(self.decoder.blocks):
                    hidden_state, kv_caches[i] = block(hidden_state, cos, sin, None, kv_caches[i], True)
                
                # Get normalized hidden state
                hidden_state_normed = self.decoder.norm(hidden_state)
                current_logits = self.decoder.head(hidden_state_normed.squeeze(1))

            # Sample next token
            probs = F.softmax(current_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, k] = next_token.squeeze(-1)
            
            # Get embedding for next iteration
            next_token_embeddings = self.decoder.token_embedding(next_token)
            
        return generated_tokens
        
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