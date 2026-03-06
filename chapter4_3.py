import torch.nn as nn
import torch

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
   
   

GPT_CONFIG_124M = {
  "vocab_size"    : 50257,   # Vocabulary size
  "context_length":  1024,   # Context length
  "emb_dim"       :   768,   # Embedding dimension
  "n_heads"       :    12,   # Number of attention heads
  "n_layers"      :    12,   # Number of layers
  "drop_rate"     :     0.1, # Drouput rate
  "qkv_bias"      :  False   # Query-Key-Value bias
}

ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)
out = ffn(x)
print(out.shape)


   
#import matplotlib.pyplot as plt
#gelu, relu = GELU(), nn.ReLU()
#
#x = torch.linspace(-3, 3, 100) # 100 sample data points in the range -3 to 3 
#y_gelu, y_relu = gelu(x), relu(x)
#plt.figure(figsize=(8, 3))
#for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
#  plt.subplot(1, 2, i)
#  plt.plot(x, y)
#  plt.title(f"{label} activation function")
#  plt.xlabel("x")
#  plt.ylabel(f"{label}(x)")
#  plt.grid(True)
#plt.tight_layout()
#plt.show()



