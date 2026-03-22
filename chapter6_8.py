from chapter6_3 import train_dataset
from chapter6_4 import model, tokenizer, device
import torch

def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()
    
    input_ids = tokenizer.encode(text)                                 # Prepares inputs to the model.
    supported_context_length = model.pos_emb.weight.shape[1]
    
    input_ids = input_ids[:min(max_length, supported_context_length)]  # Truncates sequences if they are too long
    
    input_ids += [pad_token_id] * (max_length - len(input_ids))        # Pads sequences to the longest sequence.
    
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0) # Adds batch dimension.
    
    with torch.no_grad():                                              # Models inference without gradient tracking.
        logits = model(input_tensor)[:, -1, :]                         # Logits of the last output token.
    predicted_label = torch.argmax(logits, dim=-1).item()
    
    return "spam" if predicted_label == 1 else "not spam"              # Returns the classified result.
    
text_1 = (
    "You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award."
)

print(classify_review(text_1, model, tokenizer, device, max_length=train_dataset.max_length))

text_2 = (
    "Hey, just wanted to check if we're still on"
    " for dinner tonight? Let me know!"
)

print(classify_review(text_2, model, tokenizer, device, max_length = train_dataset.max_length))

# Saving it:
torch.save(model.state_dict(), "review_classifier.pth")

# Loading it:

model_state_dict = torch.load("review_classifier.pth", map_location=device)
model.load_state_dict(model_state_dict)

