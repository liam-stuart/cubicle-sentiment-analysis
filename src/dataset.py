from collections import Counter
from torch.utils.data import Dataset
import torch
import re
from functools import lru_cache
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


STOPWORDS = nltk.corpus.stopwords.words('english')
LEMMATIZER = nltk.WordNetLemmatizer()
NON_ALPHANUM = re.compile(r'[^\w\s]')


@lru_cache(maxsize=10000)
def cached_lemmatize(word):
    return LEMMATIZER.lemmatize(word)


def clean_text(text):
    text = NON_ALPHANUM.sub('', text.lower())
    words = text.split()
    tokens = [cached_lemmatize(w) for w in words if len(w) > 1 and w not in STOPWORDS]
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
    # Just in case we scraped duplicate pages
    df.drop_duplicates(inplace=True)

    # We drop 3 star scores in order to make it easier for the models to classify positive/negative
    # By default, we filled missing scores with 0, so these get dropped too
    df = df[~(df['score'].isin([0, 3]))]

    df['product_name'] = df['product_name'].fillna('')
    df['review_text'] = df['review_text'].fillna('')
    df['review_title'] = df['review_title'].fillna('')

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
    return df


class ReviewDataset(Dataset):
    def __init__(self, vocab, sequences, labels):
        sequences["full_text"] = sequences["full_text"].apply(lambda x: text_to_sequence(x, vocab))
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq, label = self.sequences.iloc[idx], self.labels.iloc[idx]
        seq, label = seq.values[0], label.values[0]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(label, dtype=torch.float32)
