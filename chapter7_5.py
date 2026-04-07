from chapter5_3_3 import generate
from gpt_download import download_and_load_gpt2
#import My_GPT2
#from chapter04 import GPTModel
from chapter5_5_0_load_weights_into_model import load_weights_into_gpt
from chapter5_5_0_load_and_generate import assign

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

print("Model size:", model_size) #TODO!- just for checking

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
    input_text = format_input(val_data[0])
    print(input_text)
    
    

