import json
import re
import tiktoken
import time
import torch
from tqdm import tqdm
from chapter04 import GPTModel
from chapter5_1_1 import text_to_token_ids, token_ids_to_text
from chapter5_3_4 import generate
from chapter5_5_0_load_weights_into_model import load_weights_into_gpt
from chapter7_1 import train_data, val_data, test_data #TODO?~: extract train_data and the others to a separate file...?
from chapter7_1 import format_input
from chapter7_4 import train_loader, val_loader, test_loader
from chapter7_5 import train_model_simple
from gpt_download import download_and_load_gpt2

tokenizer = tiktoken.get_encoding("gpt2")
    
model_configs = {
    "gpt2-small (124M)" : { "emb_dim":  768, "n_layers": 12, "n_heads": 12 },
    "gpt2-medium (355M)": { "emb_dim": 1024, "n_layers": 24, "n_heads": 16 },
    "gpt2-large (774M)" : { "emb_dim": 1280, "n_layers": 36, "n_heads": 20 },
    "gpt2-xl (1558M)"   : { "emb_dim": 1600, "n_layers": 48, "n_heads": 25 },
}

BASE_CONFIG = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "drop_rate": 0.0,       # Dropout rate
    "qkv_bias": True        # Query-key-value bias
}

CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

settings, params = download_and_load_gpt2(
    model_size = model_size,
    models_dir = "gpt2"
)

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
#Uncomment the following two lines to use the GPU on an Apple Silicon chip.
#if torch.backends.mps.is_available():
#    device - torch.device("mps")
print("Device:", device)

model.to(device)

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

# Start listing from the book:

torch.manual_seed(123)

for entry in test_data[:3]: # Iterates over the first three test set samples
    input_text = format_input(entry)
    token_ids = generate(
        model = model,
        idx = text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens = 256,
        context_size = BASE_CONFIG["context_length"],
        eos_id = 50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    
    response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )
    
    print(input_text)
    print(f"\nCorrect response:\n>> {entry['output']}")
    print(f"\nModel response:\n>> {response_text.strip()}")
    print("-------------------------------------")

# Listing 7.9
for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
    input_text = format_input(entry)
    
    token_ids = generate(
        model = model,
        idx = text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens = 256,
        context_size = BASE_CONFIG["context_length"],
        eos_id = 50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    
    response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )
    test_data[i]["model_response"] = response_text
    
    with open("instruction-data-with-response.json", "w") as file:
      json.dump(test_data, file, indent=4) # indent for pretty-printing
      
# Verify that the responses have been correctly added to the test_set dictionary, by examining one of the entries:
if __name__ == "__main__":
    print(test_data[0])
    
# Save the (updated) model to be able to reuse it in future projects:
file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-sft.pth" # Remove whitespace and parentheses from the filename.
torch.save(model.state_dict(), file_name)
print(f"Model saved as {file_name}")

# Try to load it:
if __name__ == "__main__":
    model.load_state_dict(torch.load("gpt2-medium355M-sft.pth"))
