import tiktoken
import time
import torch
from gpt_download import download_and_load_gpt2
from chapter04 import GPTModel, generate_text_simple
from chapter5_1_1 import text_to_token_ids, token_ids_to_text
#from chapter5_2 import plot_losses #TODO?-
from chapter5_3_4 import generate
from chapter5_5_0_load_and_generate import assign
from chapter5_5_0_load_weights_into_model import load_weights_into_gpt
from chapter7_1 import format_input
from chapter7_1 import train_data, val_data, test_data #TODO?~: extract train_data and the others to a separate file...?
from chapter7_4 import train_loader, val_loader, test_loader

# Listing 7.7

#TODO?~ Do we even need this? I think I already downloaded all models...

BASE_CONFIG = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "drop_rate": 0.0,       # Dropout rate
    "qkv_bias": True        # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)" : { "emb_dim":  768, "n_layers": 12, "n_heads": 12 },
    "gpt2-medium (355M)": { "emb_dim": 1024, "n_layers": 24, "n_heads": 16 },
    "gpt2-large (774M)" : { "emb_dim": 1280, "n_layers": 36, "n_heads": 20 },
    "gpt2-xl (1558M)"   : { "emb_dim": 1600, "n_layers": 48, "n_heads": 25 },
}

CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

settings, params = download_and_load_gpt2(
    model_size = model_size,
    models_dir = "gpt2" # WAS: "gpt2"
)

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

# Trying it out:

if __name__ == "__main__":
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")
    input_text = format_input(val_data[0])
    print(input_text)
    
    token_ids = generate(
        model = model,
        idx = text_to_token_ids(input_text, tokenizer),
        max_new_tokens = 35,
        context_size = BASE_CONFIG["context_length"],
        eos_id = 50256,
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    
    response_text = generated_text[len(input_text):].strip()
    print(response_text)

# From chapter 5 (Listing 5.1, copy-pasted from the LiveBook to be sure it's correct).
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)      
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss

# From chapter 5 (Listing 5.2, copy-pasted from the LiveBook to be sure it's correct).
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches
    
# From chapter 5 (no listing #, copy-pasted from the LiveBook to be sure it's correct).
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss
    
# From chapter 5 (no listing #, copy-pasted from the LiveBook to be sure it's correct).
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
    print(decoded_text.replace("\n", " "))
    model.train()
    
# From chapter 5 (Listing 5.3, copy-pasted from the LiveBook to be sure it's correct).
def train_model_simple(model, train_loader, val_loader,
                       optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                )

        generate_and_print_sample(
            model, tokenizer, device, start_context
        )
    return train_losses, val_losses, track_tokens_seen
    
# Calculate the intial loss for the training and validation sets:
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    #Uncomment the following two lines to use the GPU on an Apple Silicon chip.
    #if torch.backends.mps.is_available():
    #    device - torch.device("mps")
    print("Device:", device)

    model.to(device)
    torch.manual_seed(123)
    
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
        
    print("Training loss:", train_loss) # Not getting the same value as in the book :-(
    print("Validation loss:", val_loss) # Not getting the same value as in the book :-(
    
if __name__ == "__main__":    
    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.00005, weight_decay = 0.1)
    num_epochs = 2

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs = num_epochs, eval_freq = 5, eval_iter = 5,
        start_context = format_input(val_data[0]), tokenizer = tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

#if __name__ == "__main__":
#    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
#    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

