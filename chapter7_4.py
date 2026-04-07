import torch
import tiktoken
from torch.utils.data import DataLoader
from chapter7_1 import train_data, val_data, test_data #TODO?~: extract train_data and the others to a separate file...?
from chapter7_3_1 import InstructionDataset #TODO?~: extract InstructionDataset to a separate file...?
from chapter7_3_2 import customized_collate_fn #TODO?~: extract custom_collate_fn and customized_collate_fn to separate files...?

tokenizer = tiktoken.get_encoding("gpt2") 

# Listing 7.6

num_workers = 0 # You can try to increase this number if parallel Python processes are supported by your operating system.
batch_size = 8

torch.manual_seed(123)

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size = batch_size,
    collate_fn = customized_collate_fn,
    shuffle = True,
    drop_last = True,
    num_workers = num_workers
)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size = batch_size,
    collate_fn = customized_collate_fn,
    shuffle = False,
    drop_last = False,
    num_workers = num_workers
)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size = batch_size,
    collate_fn = customized_collate_fn,
    shuffle = False,
    drop_last = False,
    num_workers = num_workers
)

# Try it out:
if __name__ == "__main__":
    print("Train loader")
    for inputs, targets in train_loader:
        print(inputs.shape, targets.shape)
        
        

