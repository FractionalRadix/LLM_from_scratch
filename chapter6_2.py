import pandas as pd

def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]                            # Counts the instances of "spam"
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123) # Randomly samples "ham" instances to match the number of "spam" instances
    balanced_df = pd.conccat([ham_subset, df[df["Label"] == "spam"]])        # Combines ham subset with "spam"
    return balanced_df

extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"
df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
print(df)

balanced_df = create_balanced_dataset(df)
print(balanced_df["Label"].value_counts())
