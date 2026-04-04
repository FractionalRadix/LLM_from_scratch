import tiktoken
import torch
from gpt_download import download_and_load_gpt2
from chapter04 import GPTModel, generate_text_simple
from My_GPT2 import load_settings_and_params, assign, load_weights_into_gpt, text_to_token_ids, token_ids_to_text
from chapter6_3 import train_loader, val_loader, test_loader

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

probas = torch.softmax(outputs[:, -1, :], dim=-1)
label = torch.argmax(probas)
print("Class label:", label.item())

# We don't even need the softmax function in this case, because the largest outputs directly correspond to the highest probability scores.
logits = outputs[:, -1, :]
label = torch.argmax(logits)
print("Class label:", label.item())

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0
    
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
        
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            
            with torch.no_grad():
                logits = model(input_batch)[:, -1, :] # Logits of last output token
            predicted_labels = torch.argmax(logits, dim=-1)
            
            num_examples += predicted_labels.shape[0]
            correct_predictions += (
                (predicted_labels == target_batch).sum().item()
            )
        else:
            break
            
    return correct_predictions / num_examples
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

torch.manual_seed(123)
train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)
val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10)
test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10)

print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :] # Logits of last output token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss
    
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader)==0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))  # Ensures number of batches doens't exceed batches in data loader
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
            
    return total_loss / num_batches
    
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device, num_batches = 5)
    val_loss   = calc_loss_loader(val_loader,   model, device, num_batches = 5)
    test_loss  = calc_loss_loader(test_loader,  model, device, num_batches = 5)
print(f"Training loss: {train_loss:.3f}")
print(f"Validation loss: {val_loss:.3f}")
print(f"Test loss: {test_loss:.3f}")

def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter):
    train_losses, val_losses, train_accs, val_accs = [], [], [], [] # Initialize lists to track losses and examples seen
    examples_seen, global_step = 0, -1
    
    for epoch in range(num_epochs):                                 # Main trainig loop
        model.train()                                               # Sets model to training mode
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()                                   # Resets loss gradients from the previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()                                         # Calculates loss gradients
            optimizer.step()                                        # Updates model weights using loss gradients
            examples_seen += input_batch.shape[0]                   # New (since earlier chapter, SB): tracks examples instead of tokens
            global_step += 1
            
            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}):"
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                )
                
        # Calculates accuracy after each epoch                
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy   = calc_accuracy_loader(val_loader,   model, device, num_batches=eval_iter)
        
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
        
    return train_losses, val_losses, train_accs, val_accs, examples_seen
    
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss   = calc_loss_loader(val_loader,   model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss
    
import time

start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
num_epochs = 5

train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50,
        eval_iter=5
    )

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")


