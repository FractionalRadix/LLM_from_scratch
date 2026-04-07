import torch
#from torch.utils.data import Dataset

from chapter7_1 import format_input

# Listing 7.5

# Token 50256 is "<|endoftext|>"        
def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device="cpu"):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []
    
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        
        padded = (new_item + [pad_token_id] * (batch_max_length - len(new_item))) # Pads sequences to max_length.
        inputs = torch.tensor(padded[:-1]) # Truncates the last token for inputs.
        targets = torch.tensor(padded[1:]) # Shifts +1 to the right for targets.
        
        # Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index
            
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
            
        inputs_lst.append(inputs)
        targets_lst.append(targets)
        
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor
    
# Try it out.
if __name__ == "__main__":
    inputs_1 = [0, 1, 2, 3, 4]
    inputs_2 = [5, 6]
    inputs_3 = [7, 8, 9]    
    batch = (inputs_1, inputs_2, inputs_3)

    inputs, targets = custom_collate_fn(batch)
    print(inputs)
    print(targets)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
#Uncomment the following two lines to use the GPU on an Apple Silicon chip.
#if torch.backends.mps.is_available():
#    device - torch.device("mps")
print("Device:", device)
    
from functools import partial

customized_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=1024
)
