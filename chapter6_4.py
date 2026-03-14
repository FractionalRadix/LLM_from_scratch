import tiktoken
import torch
from gpt_download import download_and_load_gpt2
from chapter04 import GPTModel, generate_text_simple
from My_GPT2 import load_settings_and_params, assign, load_weights_into_gpt, text_to_token_ids, token_ids_to_text

CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 1.0,
    "qkv_bias": True
}    

#TODO?- Just import My_GPT2.py where these are also defined?
model_configs = {
    "gpt2-small (124M)" : { "emb_dim":  768, "n_layers": 12, "n_heads": 12 },
    "gpt2-medium (355M)": { "emb_dim": 1024, "n_layers": 24, "n_heads": 16 },
    "gpt2-large (774M)" : { "emb_dim": 1280, "n_layers": 36, "n_heads": 20 },
    "gpt2-xl (1558M)"   : { "emb_dim": 1600, "n_layers": 48, "n_heads": 25 },
}
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
#print("model_size==", model_size)
#TODO!~ settings, params - get them from the normal gpt2 directory... no need to load them from the net!
#settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2x")
settings, params = load_settings_and_params(model_size, models_dir="gpt2")
model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

text_1 = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_1, tokenizer),
    max_new_tokens=15,
    context_size=BASE_CONFIG["context_length"]
)
print(token_ids_to_text(token_ids, tokenizer))

text_2 = (
    "Is the following text 'spam'? Answer with 'yes' or 'no':"
    " 'You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award.'"
)
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_2, tokenizer),
    max_new_tokens=23,
    context_size=BASE_CONFIG["context_length"]
)
print(token_ids_to_text(token_ids, tokenizer))


# My own experiment, to see if it keeps repeating the last sentence when I change things.
#text_3 = (
#    "Is 'It was a dark and stormy night' a good start for a novel? "
#)
#token_ids = generate_text_simple(
#    model=model,
#    idx=text_to_token_ids(text_3, tokenizer),
#    max_new_tokens=250,
#    context_size=BASE_CONFIG["context_length"]
#)
#print(token_ids_to_text(token_ids, tokenizer))

print(model)

# Freeze the model, to make all layers nontrainable:
for param in model.parameters():
    param.requires_grad_ = False
    
# Replace the output layer (model.out_head), which originally maps the layer inputs to 50257 dimensions (the size of the vocabulary), to 2 - for 'yes' and 'no'.
# The new `model.out_head` output layer has its `requires_grad` attribute set to `True` by default.
torch.manual_seed(123)
num_classes = 2
model.out_head = torch.nn.Linear(
    in_features=BASE_CONFIG["emb_dim"],
    out_features=num_classes
)
#print(model.out_head.requires_grad_)

# Fine-tuning multiple layers can notably improve predictive performance. 
# Therefore we make the final `LayerNorm` and the last transformer block trainable as well.
for param in model.trf_blocks[-1].parameters():
    param.requires_grad_=True
for param in model.final_norm.parameters():
    param.requires_grad_=True
    
inputs = tokenizer.encode("Do you have time")
inputs = torch.tensor(inputs).unsqueeze(0)
print("Inputs:", inputs)
print("Inputs dimensions:", inputs.shape) # shape: (batch_size, num_tokens)

with torch.no_grad():
    outputs = model(inputs)
print("Outputs:\n", outputs)
print("Outputs dimensions:", outputs.shape)

print("Last output token:", outputs[:, -1, :])

