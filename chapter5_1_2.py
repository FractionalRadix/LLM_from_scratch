import torch
from chapter04 import GPTModel, generate_text_simple
from chapter5_1_1 import text_to_token_ids, token_ids_to_text
import tiktoken

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

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                       [   40, 1107,  588]])  #  "I really like"]
                       
targets = torch.tensor([[3626, 6100, 345 ],   # ["effort moves you",
                        [1107,  588, 11311]]) #  "really like chocolate"]
                        
with torch.no_grad():
   logits = model(inputs)
   
probas = torch.softmax(logits, dim=-1)
print(probas.shape)

token_ids = torch.argmax(probas, dim=-1, keepdim=True)
print("Token IDs:\n", token_ids) # In die eerdere was het [[  .. , .., .. ... .. ]]  met shape (1,14), VOOR de squeeze. Nu is het... shape (2,3,1) .

print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
print(f"Outputs batch 1:"
    f"{token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 1:", target_probas_1)

text_idx = 1	
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 2:", target_probas_2)

log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)

avg_log_probas = torch.mean(log_probas)
print(avg_log_probas)

neg_avg_log_probas = avg_log_probas * -1
print(neg_avg_log_probas)

print("Logits shape:", logits.shape)
print("Targets shape:", targets.shape)

# Remember:
# * `targets` are the token ID's that we want the LLM to generate.
# * `logits` contain the unscaled model outputs before they enter the softmax function to obtain the probability scores.
logits_flat = logits.flatten(0, 1)
targets_flat =  targets.flatten()
print("Flattened logits:", logits_flat.shape)
print("Flattened targets:", targets_flat.shape)

loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print(loss) # Same as neg_avg_log_probas - meaning torch can save us the intermediate steps.

# Added by me, from the sidebar in the book.
perplexity = torch.exp(loss) # Perplexity of about N means the model being unsure about which among N tokens in the vocabulary to generate as the next one.
print(perplexity)


