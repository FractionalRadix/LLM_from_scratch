import tiktoken
import torch
from chapter04 import GPTModel

GPT_CONFIG_124M = {
  "vocab_size"    : 50257,   # Vocabulary size
  "context_length":   256,   # Context length - shortened from the 1024 that we used in Chapter 04.
  "emb_dim"       :   768,   # Embedding dimension
  "n_heads"       :    12,   # Number of attention heads
  "n_layers"      :    12,   # Number of layers
  "drop_rate"     :     0.1, # Drouput rate. It's possible and common to set dropout to 0.
  "qkv_bias"      :  False   # Query-Key-Value bias
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")

file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()
    
total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters:", total_characters)
print("Tokens:", total_tokens)

train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]  # Training data
val_data = text_data[split_idx:]    # Validation data

from chapter02 import create_dataloader_v1

torch.manual_seed(123)

train_loader = create_dataloader_v1(
    train_data,
    batch_size = 2,
    max_length = GPT_CONFIG_124M["context_length"],
    stride = GPT_CONFIG_124M["context_length"],
    drop_last = True,
    shuffle = True,
    num_workers = 0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size = 2,
    max_length = GPT_CONFIG_124M["context_length"],
    stride = GPT_CONFIG_124M["context_length"],
    drop_last = False,
    shuffle = False,
    num_workers = 0
)

print("Train loader:")
for x, y in train_loader:
    print(x.shape, y.shape)
    
print("\nValidation loader:")
for x, y in val_loader:
   print(x.shape, y.shape)

def calc_loss_batch(input_batch, target_batch, model, device):
    # The transfer to a given device allows us to transfer the data to a GPU.
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss
    
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches == None:
        num_batches = len(data_loader) # Iteratives over all batches if no fixed num_batches is specfied.
    else:
        num_batches = min(num_batches, len(data_loader)) # Reduces the number of batches to match the total number of batches in the data loader
                                                         # if num_bathes exceeds the number of batches in the data loader.
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item() # Sums loss for each batch.
        else:
            break
    return total_loss / num_batches # Averages the loss over all batches.
    
# If you have a machine with a CUDA-supported GPU, the LLM will train on the GPU without making any changes to the code.    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
with torch.no_grad(): # Disables gradient tracking for efficiency because we are not training yet.
    # Via the "device" setting, we ensure the data is loaded onto the same device as the LLM model.
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device) 
print("Training loss:", train_loss)
print("Validation loss:", val_loss)
