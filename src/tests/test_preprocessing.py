import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from preprocessing import clean_text, build_vocab, text_to_sequence, preprocess_dataframe, create_datasets


def test_clean_text():
    test_review = "Puzzle's corner-cutting excellent 100% RECOMMEND 😉"
    text = clean_text(test_review)
    assert text == "puzzle cornercutting excellent recommend"


def test_build_vocab():
    test_df = pd.DataFrame({"full_text": ["word word other"]})
    vocab, vocab_size = build_vocab(test_df)
    assert vocab_size == 3
    assert vocab["<PAD>"] == 0
    assert vocab["<UNK>"] == 1
    assert vocab["word"] == 2


def test_text_to_sequence():
    test_df = pd.DataFrame({"full_text": ["word word other"]})
    vocab, vocab_size = build_vocab(test_df)
    test_df["full_text"] = test_df["full_text"].apply(lambda x: text_to_sequence(x, vocab))
    sequence = test_df.loc[0, "full_text"]
    assert sequence == [2, 2, 1]


def test_preprocess_dataframe():
    test_df = pd.DataFrame({
        "product_name": ["name1", "name2", "name2", "name4", None],
        "review_title": ["title1", "keep2", "keep2", "title4", None],
        "review_text": [" Name1  ", "review2", "review2", "review4", None],
        "score": [5, 2, 2, 0, 4]
    })
    test_df = preprocess_dataframe(test_df)
    assert len(test_df) == 1
    assert test_df.loc[0, "full_text"] == "keep review"
    assert test_df.loc[0, "is_positive"] == 0


def test_create_datasets():
    test_df = pd.DataFrame({
        "full_text": (["word"] * 50) + (["word other"] * 50),
        "is_positive": ([1] * 80) + ([0] * 20)
    })
    text = test_df[["full_text"]]
    labels = test_df[["is_positive"]]
    train_df, val_df, train_labels, val_labels = train_test_split(text, labels, random_state=42, test_size=0.2,
                                                                  stratify=labels)
    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
        "word": 2,
        "other": 3
    }

    train_dataset, val_dataset = create_datasets(vocab, train_df, train_labels, val_df, val_labels)
    assert train_dataset.__len__() == 80
    assert val_dataset.__len__() == 20
    assert train_dataset.__getitem__(0)[0].shape == torch.Size([2])
    assert val_dataset.__getitem__(0)[1].shape == torch.Size([1])
