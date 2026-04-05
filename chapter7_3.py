import torch
from torch.utils.data import Dataset

from chapter7_1 import format_input

# Listing 7.4

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))
            
    def __getitem__(self, index):
        return self.encoded_texts(index)
        
    def __len__(self):
        return len(self.data)
        
# Token 50256 is "<|endoftext|>"        
def custom_collate_draft_1(batch, pad_token_id = 50256, device="cpu"):
    batch_max_length = max(len(item) + 1 for item in batch) # Finds [1 + the length of] the longest sequence in the batch.
    inputs_lst = []
    
    # Pads and prepares inputs.
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (new_item + [pad_token_id] * (batch_max_length - len(new_item)))
        inputs = torch.tensor(padded[:-1]) # Removes extra padded token added earlier.
        inputs_lst.append(inputs)
        
    inputs_tensor = torch.stack(inputs_lst).to(device) # Converts the list of inputs to a tensor and transfers it to the target device.
    return inputs_tensor
    
# Try it out:
if __name__ == "__main__":
    inputs_1 = [0, 1, 2, 3, 4]
    inputs_2 = [5, 6]
    inputs_3 = [7, 8, 9]
    
    batch = (inputs_1, inputs_2, inputs_3)
    
    print(custom_collate_draft_1(batch))

