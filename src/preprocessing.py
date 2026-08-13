import re
from collections import Counter
from functools import lru_cache
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from torch.nn.utils.rnn import pad_sequence
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


STOPWORDS = nltk.corpus.stopwords.words('english')
LEMMATIZER = nltk.WordNetLemmatizer()
PATTERN = re.compile(r'[^\w\s]|\d+')


@lru_cache(maxsize=10000)
def cached_lemmatize(word):
    return LEMMATIZER.lemmatize(word)


def clean_text(text: str) -> str:
    text = PATTERN.sub('', text.lower())
    words = text.split()
    tokens = [cached_lemmatize(w) for w in words if len(w) > 1 and w not in STOPWORDS]
    return " ".join(tokens)


def build_vocab(df: pd.DataFrame) -> tuple[dict[str, int], int]:
    all_words = " ".join(df['full_text']).split()
    word_counts = Counter(all_words)
    valid_words = [word for word, count in word_counts.items() if count >= 2]
    vocab = {word: i + 2 for i, word in enumerate(valid_words)}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    vocab_size = max(vocab.values()) + 1
    return vocab, vocab_size


def text_to_sequence(text: str, vocab: dict[str, int]) -> list[int]:
    return [vocab.get(word, vocab["<UNK>"]) for word in text.split()][:64]


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Just in case we scraped duplicate pages
    df.drop_duplicates(inplace=True)

    # We drop 3 star scores in order to make it easier for the models to classify positive/negative
    # By default, we filled missing scores with 0, so these get dropped too
    df = df[~(df['score'].isin([0, 3]))]

    # Many reviews don't have titles, a rare few also have no text
    # None in the default output.csv have missing titles, but included just in case
    for col in ["product_name", "review_title", "review_text"]:
        df.loc[df[col].isna(), col] = ""

    # A fair amount of reviews just have the product name as the review text, likely due to a migration
    # We just drop these reviews as their text is not correlated with the review score
    prod_names = [x.strip().lower().replace(" ", "") for x in df['product_name']]
    rev_texts = [x.strip().lower().replace(" ", "") for x in df['review_text']]
    mask = [p != r for p, r in zip(prod_names, rev_texts)]
    df = df[mask]

    df['full_text'] = df['review_title'] + ' ' + df['review_text']
    df['is_positive'] = (df['score'] > 2).astype(int)

    df.drop(['product_name', 'review_title', 'review_text', 'score'], axis=1, inplace=True)
    df["full_text"] = df["full_text"].apply(lambda x: clean_text(x))
    df = df[df["full_text"] != ""]
    df.reset_index(drop=True, inplace=True)
    return df


def create_datasets(vocab: dict[str, int], train_df: pd.DataFrame, train_labels: pd.DataFrame,
                    val_df: pd.DataFrame, val_labels: pd.DataFrame) -> tuple[TensorDataset, TensorDataset]:
    train_df["full_text"] = train_df["full_text"].apply(lambda x: text_to_sequence(x, vocab))
    val_df["full_text"] = val_df["full_text"].apply(lambda x: text_to_sequence(x, vocab))

    train_sequences = [torch.tensor(x, dtype=torch.long) for x in train_df["full_text"]]
    val_sequences = [torch.tensor(x, dtype=torch.long) for x in val_df["full_text"]]

    train_df_tensor = pad_sequence(train_sequences, batch_first=True, padding_value=0)
    train_labels_tensor = torch.tensor(train_labels.values, dtype=torch.float32)
    val_df_tensor = pad_sequence(val_sequences, batch_first=True, padding_value=0)
    val_labels_tensor = torch.tensor(val_labels.values, dtype=torch.float32)

    train_dataset = TensorDataset(train_df_tensor, train_labels_tensor)
    val_dataset = TensorDataset(val_df_tensor, val_labels_tensor)

    return train_dataset, val_dataset
