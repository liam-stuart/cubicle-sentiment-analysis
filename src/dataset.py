from sklearn.model_selection import train_test_split
from collections import Counter
from torch.utils.data import Dataset
import torch
import re
import os
import json
import pandas as pd
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


def clean_text(text):
    stopwords = nltk.corpus.stopwords.words('english')
    wl = nltk.WordNetLemmatizer()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.split()
    tokens = []
    for word in text:
        word_lower = word.lower()
        if len(word_lower) > 1 and word_lower not in stopwords:
            tokens.append(wl.lemmatize(word_lower))
    return " ".join(tokens)


def build_vocab(df):
    all_words = " ".join(df['full_text']).split()
    word_counts = Counter(all_words)
    valid_words = [word for word, count in word_counts.items() if count >= 2]
    vocab = {word: i + 2 for i, word in enumerate(valid_words)}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab


def text_to_sequence(text, vocab):
    return [vocab.get(word, vocab["<UNK>"]) for word in text.split()][:64]


def preprocess_dataframe(df):
    df['product_name'] = df['product_name'].fillna('')
    df['review_text'] = df['review_text'].fillna('')
    df['review_title'] = df['review_title'].fillna('')
    # A fair amount of reviews just have the product name as the review text, likely due to a migration
    # We just drop these reviews as their text is not correlated with the review score
    df['product_name_combined'] = df['product_name'].apply(lambda x: x.strip().lower().replace(" ", ""))
    df['review_text_combined'] = df['review_text'].apply(lambda x: x.strip().lower().replace(" ", ""))
    df = df[~(df['product_name_combined'] == df['review_text_combined'])]
    # Just in case we scraped duplicate pages
    df.drop_duplicates(inplace=True)
    df['full_text'] = df['review_title'] + ' ' + df['review_text']
    # We drop 3 star scores in order to make it easier for the models to classify positive/negative
    df = df[~(df['score'].isin([0, 3]))]
    df['is_positive'] = (df['score'] > 2).astype(int)

    df.drop(['product_name', 'review_title', 'review_text', 'product_name_combined',
             'review_text_combined', 'score'], axis=1, inplace=True)
    df["full_text"] = df["full_text"].apply(lambda x: clean_text(x))
    df = df[df["full_text"] != ""]
    return df


class ReviewDataset(Dataset):
    def __init__(self, vocab, train=True):
        if train:
            path_to_data = "train"
        else:
            path_to_data = "val"

        text_path = f"{path_to_data}/text.csv"
        label_path = f"{path_to_data}/labels.csv"
        text = pd.read_csv(text_path)
        text["full_text"] = text["full_text"].apply(lambda x: text_to_sequence(x, vocab))
        self.sequences = text
        self.labels = pd.read_csv(label_path)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq, label = self.sequences.iloc[idx], self.labels.iloc[idx]
        seq, label = seq.values[0], label.values[0]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(label, dtype=torch.float32)


if __name__ == "__main__":
    df = pd.read_csv("output.csv")
    print('Processing data...')
    df = preprocess_dataframe(df)

    text = df[["full_text"]]
    labels = df[["is_positive"]]
    train_df, val_df, train_labels, val_labels = train_test_split(text, labels, test_size=0.2, stratify=labels)

    vocab = build_vocab(train_df)
    with open("vocab.json", "w") as f:
        json.dump(vocab, f)

    os.makedirs("train/", exist_ok=True)
    os.makedirs("val/", exist_ok=True)
    train_df.to_csv("train/text.csv", index=False)
    train_labels.to_csv("train/labels.csv", index=False)
    val_df.to_csv("val/text.csv", index=False)
    val_labels.to_csv("val/labels.csv", index=False)
    print("Data processed successfully!")
