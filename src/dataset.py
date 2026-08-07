import torch
from torch.utils.data import Dataset
from collections import Counter
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')


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
    all_words = " ".join(df['text_combined']).split()
    word_counts = Counter(all_words)
    valid_words = [word for word, count in word_counts.items() if count >= 2]
    vocab = {word: i + 2 for i, word in enumerate(valid_words)}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab


def text_to_sequence(text, vocab):
    return [vocab.get(word, vocab["<UNK>"]) for word in text.split()][:256]


def preprocess_dataframe(df):
    df['review_text'] = df['review_text'].fillna('')
    df['review_title'] = df['review_title'].fillna('')
    # A fair amount of reviews just have the product name as the review text, likely due to a migration.
    # We just drop these reviews as their text is not correlated with the review score.
    df['product_name_combined'] = df['product_name'].apply(lambda x: x.strip().lower().replace(" ", ""))
    df['review_text_combined'] = df['review_text'].apply(lambda x: x.strip().lower().replace(" ", ""))
    df = df[~(df['product_name_combined'] == df['review_text_combined'])]
    # Just in case we scraped duplicate pages.
    df.drop_duplicates(inplace=True)
    # We drop 3 start scores in order to make it easier for the models to classify positive/negative.
    df = df[~(df['score'] == 3)]
    df['is_positive'] = (df['score'] > 2).astype(int)

    df['text_combined'] = df['review_title'] + ' ' + df['review_text']
    df.drop(['review_title', 'review_text', 'product_name_combined',
             'review_text_combined', 'score'], axis=1, inplace=True)
    df["text_combined"] = df["text_combined"].apply(lambda x: clean_text(x))
    return df


class ReviewDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float32)
