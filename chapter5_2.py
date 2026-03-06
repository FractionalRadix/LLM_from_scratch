import torch
import tiktoken
from chapter02 import create_dataloader_v1
from chapter04 import GPTModel
from chapter04 import generate_text_simple
#from chapter5_1_1 import text_to_token_ids, token_ids_to_text

def train_model_simple(model, train_loader, val_loader,
                       optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], [] # Initialize lists to track losses and tokens seen.
    tokens_seen, global_step = 0, -1
       
    for epoch in range(num_epochs):                          # Starts the main training loop.
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()                            # Resets loss gradients from the previous batch iteration.
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()                                  # Calculates loss gradients.
            optimizer.step()                                 # Updates model weights using loss gradients.
            tokens_seen += input_batch.numel()
            global_step += 1
               
            if global_step % eval_freq == 0:                 # Optional evaluation step
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                )
                   
        generate_and_print_sample(model, tokenizer, device, start_context) # Prints a sample after each epoch
    return train_losses, val_losses, track_tokens_seen
    
    
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()          # Dropout is disabled during evaluation for stable, reproducible results.
    with torch.no_grad(): # Disables gradient tracking, which is not required during evaluation, to reduce the computational overhead
         train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
         val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss
    
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " ")) # Compact print format
    model.train()

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # .unsqueeze(0) adds the batch dimension.
    return encoded_tensor
    
def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # Removes batch dimension.
    return tokenizer.decode(flat.tolist())


GPT_CONFIG_124M = {
  "vocab_size"    : 50257,   # Vocabulary size
  "context_length":   256,   # Context length - shortened from the 1024 that we used in Chapter 04.
  "emb_dim"       :   768,   # Embedding dimension
  "n_heads"       :    12,   # Number of attention heads
  "n_layers"      :    12,   # Number of layers
  "drop_rate"     :     0.1, # Dropout rate. It's possible and common to set dropout to 0.
  "qkv_bias"      :  False   # Query-Key-Value bias
}

#### From chapter5_1_3.py:

tokenizer = tiktoken.get_encoding("gpt2")

file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()
    
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]  # Training data
val_data = text_data[split_idx:]    # Validation data

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
        num_batches = len(data_loader) # Iterates over all batches if no fixed num_batches is specified.
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
#model.to(device)

### End of imports from chapter5_1_3.py

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(
    model.parameters(), # The .parameters() method returns all trainable weight parameters of the model.
    lr=0.0004, weight_decay=0.1
)
num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()                            # Creates a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0) # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()

#epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
#plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

