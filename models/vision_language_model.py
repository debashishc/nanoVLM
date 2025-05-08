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

        logits = self.decoder(combined_embd, attention_mask) # Not logits yet, but easier to return like this

        loss = None
        if targets is not None:
            # Only use the token part of the logits for loss computation
            logits = self.decoder.head(logits)
            logits = logits[:, image_embd.size(1):, :]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)

        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, image, attention_mask=None, max_new_tokens=5):
        # Process image through vision encoder and projection
        image_embd = self.vision_encoder(image)
        image_embd = self.MP(image_embd)
        
        # Embed initial tokens
        token_embd = self.decoder.token_embedding(input_ids)
        
        # Concatenate image embeddings with token embeddings
        combined_embd = torch.cat((image_embd, token_embd), dim=1)

        batch_size = image_embd.size(0)
        img_seq_len = image_embd.size(1)
        
        # Adjust attention mask to account for image tokens
        if attention_mask is not None:
            # Create mask of 1s for image tokens (all image tokens should be attended to)
            image_attention_mask = torch.ones((batch_size, img_seq_len), device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat((image_attention_mask, attention_mask), dim=1)
        
        # Initialize KV cache for each block in the decoder
        kv_caches = [None] * len(self.decoder.blocks)
        
        # Process the initial sequence to populate the KV cache
        position_ids = torch.arange(combined_embd.size(1), device=combined_embd.device).unsqueeze(0).expand(batch_size, -1)
        cos, sin = self.decoder.rotary_embd(position_ids)
        
        # Forward pass through each block to initialize the KV cache
        current_outputs = combined_embd
        for i, block in enumerate(self.decoder.blocks):
            current_outputs, kv_caches[i] = block(current_outputs, cos, sin, attention_mask, None, True)
        
        current_outputs = self.decoder.norm(current_outputs)
        
        # Generate tokens using the cached KV values
        generated_tokens = torch.zeros((batch_size, max_new_tokens), device=input_ids.device, dtype=input_ids.dtype)
        
        seq_len = combined_embd.size(1)
        
        for i in range(max_new_tokens):
            # Get the last token's position for rotary embeddings
            position_id = torch.full((batch_size, 1), seq_len + i, device=combined_embd.device)
            cos, sin = self.decoder.rotary_embd(position_id)
            
            # Only process the last token with cached KV values
            last_token = current_outputs[:, -1:, :]
            
            # Forward pass through each block, using and updating the KV cache
            for j, block in enumerate(self.decoder.blocks):
                last_token, kv_caches[j] = block(last_token, cos, sin, None, kv_caches[j], True)
            
            # Apply the final normalization
            last_token = self.decoder.norm(last_token)
            
            # Convert to logits if needed
            if not self.decoder.lm_use_tokens:
                last_token_logits = self.decoder.head(last_token)
            else:
                last_token_logits = last_token

            probs = torch.softmax(last_token_logits.squeeze(1), dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
                
            generated_tokens[:, i] = next_token.squeeze(-1)
            
            # Get the embedding for the next token and append to current outputs for the next iteration
            next_embd = self.decoder.token_embedding(next_token)
            current_outputs = torch.cat([current_outputs, next_embd], dim=1)
            
            # If we had attention mask, we need to extend it for the new token
            if attention_mask is not None:
                attention_mask = torch.cat((attention_mask, torch.ones((batch_size, 1), device=attention_mask.device)), dim=1)
        
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