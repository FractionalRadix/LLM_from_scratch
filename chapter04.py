import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
  def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
    super().__init__()
    #assert (d_out % num_heads == 0, "d_out must be divisible by num_heads")
    self.d_out = d_out
    self.num_heads = num_heads
    self.head_dim = d_out // num_heads # Reduces the projection dim to match the desired output dim.
    self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.out_proj = nn.Linear(d_out, d_out) # Use a Linear layer to combine head outputs.
    self.dropout = nn.Dropout(dropout)
    self.register_buffer(
      "mask",
      torch.triu(torch.ones(context_length, context_length), diagonal=1)
    )
    
  def forward(self, x):
    b, num_tokens, d_in = x.shape
    # Tensor shape: (b, num_tokens, d_out)
    keys = self.W_key(x)
    queries = self.W_query(x)
    values = self.W_value(x)
        
    # We implicitly split the matrix by adding a num_heads dimension. 
    # Then we unroll the last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim).
    keys    = keys.view(b, num_tokens, self.num_heads, self.head_dim)
    values  = values.view(b, num_tokens, self.num_heads, self.head_dim)
    queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
    
    # Transpose from shape (b, num_tokens, num_heads, head_dim) to (b, num_heads, num_tokens, head_dim).
    keys = keys.transpose(1, 2)
    values = values.transpose(1, 2)
    queries = queries.transpose(1, 2)
    
    # Compute dot product for each head.
    attn_scores = queries @ keys.transpose(2, 3)
    
    # Masks truncated to the number of tokens.
    mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
    
    # Use the mask to fill the attention scores.
    attn_scores.masked_fill_(mask_bool, -torch.inf)
    
    attn_weights = torch.softmax( attn_scores / keys.shape[-1]**0.5, dim=-1 )
    attn_weights = self.dropout(attn_weights)
    
    context_vec = (attn_weights @ values).transpose(1, 2) # Tensor shape: (b, num_tokens, num_heads, head_dim)
    # Combine heads, where self.d_out = self.num_heads * self.head_dim
    context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out) 
    # Add an optional linear projection.
    context_vec = self.out_proj(context_vec)
    
    return context_vec

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
     
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]  # Crops current context if it exceeds the supported context size (uses last ... tokens)
        with torch.no_grad():
            logits = model(idx_cond)
            
        logits = logits[:, -1, :]                             # Focuses only on the last time step, so that (batch, n_token, vocab_size) becomes (batch, vocab_size).
        probas = torch.softmax(logits, dim=-1)                # `probas` has shape (batch, vocab_size).
        idx_next = torch.argmax(probas, dim=-1, keepdim=True) # `idx_next` has shape (batch, 1).
        idx = torch.cat((idx, idx_next), dim=1)               # Appends sampled index to the running sequence, where` idx` has shape (batch, n_tokens+1).
        
    return idx

