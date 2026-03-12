import pandas as pd
from pathlib import Path

def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]                            # Counts the instances of "spam"
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123) # Randomly samples "ham" instances to match the number of "spam" instances
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])        # Combines ham subset with "spam"
    return balanced_df

extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"
df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
print(df)

balanced_df = create_balanced_dataset(df)
print(balanced_df["Label"].value_counts())

balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

def random_split(df, train_frac, validation_frac):
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)    # Shuffles the entire DataFrame.
    train_end = int(len(df) * train_frac)                              # Calculates split indices
    validation_end = train_end + int(len(df) * validation_frac)
    
    train_df = df[:train_end]                                          # Splits the DataFrame
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]
    
    return train_df, validation_df, test_df
    
train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1) # Test size is implied to be 0.2 as the remainder.

train_df.to_csv("train.csv", index=None)
validation_df.to_csv("validation.csv", index=None)
test_df.to_csv("test.csv", index=None)


