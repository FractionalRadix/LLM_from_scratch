import torch
import torch.nn as nn
from chapter03 import MultiHeadAttention
import tiktoken

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential( *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        
        # The device setting wil allow us to train the model on a CPU or GPU, depending on which device the input data sits on.
        
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
     
     
GPT_CONFIG_124M = {
  "vocab_size"    : 50257,   # Vocabulary size
  "context_length":  1024,   # Context length
  "emb_dim"       :   768,   # Embedding dimension
  "n_heads"       :    12,   # Number of attention heads
  "n_layers"      :    12,   # Number of layers
  "drop_rate"     :     0.1, # Dropout rate
  "qkv_bias"      :  False   # Query-Key-Value bias
}

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh( torch.sqrt(torch.tensor (2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))
        
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )            
    
    def forward(self, x):
        return self.layers(x)

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5                                   # Epsilon; added to variance to prevent division by zero during normalization.
        self.scale = nn.Parameter(torch.ones(emb_dim))    # Learnable parameter.
        self.shift = nn.Parameter(torch.zeros(emb_dim))   # Learnable parameter.
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # unbiased = False: not using Bessel's correction since n is large (n, n-1 .. negligible).
        norm_x = (x-mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length = cfg["context_length"],
            num_heads = cfg["n_heads"],
            dropout = cfg["drop_rate"],
            qkv_bias = cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
        
    def forward(self, x):
    
        shortcut = x              # Shortcut connection for attention block.
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut          # Add the original input back.
        
        shortcut = x              # Shortcut connection for feed forward block.
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut          # Add the original input back.
        
        return x


#TODO!~ Does not give the same numbers as the book! 
# Unclear why; copy-pasting the GPTModel class from the official GitHub (at https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/01_main-chapter-code/ch04.ipynb )
# did NOT make a difference. So, maybe something was done to the batch in-between?
# Code from earlier paragraphs DID give the same results as the book, but of course there could still be a subtle error in there.

batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
tokenizer = tiktoken.get_encoding("gpt2")
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)

