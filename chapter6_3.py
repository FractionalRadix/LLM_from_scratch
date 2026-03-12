import tiktoken
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]   # Pretokenizes texts
        ]
        
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            self.encoded_texts = [                                  # Truncates sequences if they are longer than max_length
                encoded_text[:self.max_length] for encoded_text in self.encoded_texts
            ]
            
        self.encoded_texts = [                                      # Pad sequences to the longest sequence
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text)) for encoded_text in self.encoded_texts
        ]
        
    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )
        
    def __len__(self):
        return len(self.data)
                    
    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length

tokenizer = tiktoken.get_encoding("gpt2")
print("Let's find the token value for the <|endoftext|> special token:")
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

train_dataset = SpamDataset(csv_file="train.csv", max_length=None, tokenizer=tokenizer)
print("Maximum item length in the training dataset: ", train_dataset.max_length)
val_dataset = SpamDataset(csv_file="validation.csv", max_length = train_dataset.max_length, tokenizer=tokenizer)
test_dataset = SpamDataset(csv_file="test.csv", max_length = train_dataset.max_length, tokenizer=tokenizer)

num_workers = 0  # This setting ensures compatibility with most computers
batch_size = 8
torch.manual_seed(123)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False)

for input_batch, target_batch in train_loader:
   pass
print("Input batch dimensions:", input_batch.shape)
print("Label batch dimensions:", target_batch.shape) 

print(f"{len(train_loader)} training batches")
print(f"{len(val_loader)} validation batches")
print(f"{len(test_loader)} test batches")

