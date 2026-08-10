import logging
import time
import streamlit as st
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from model import Model
from preprocessing import clean_text, build_vocab, text_to_sequence, preprocess_dataframe, create_datasets
from utils import load_checkpoint
from train import train_model


logger = logging.getLogger("cubicle-app")
logging.basicConfig(level=logging.ERROR)
device = "cuda" if torch.cuda.is_available() else "cpu"

if "vocab" in st.session_state:
    vocab = st.session_state.vocab
    vocab_size = st.session_state.vocab_size
    train_dataset = st.session_state.train_dataset
    val_dataset = st.session_state.val_dataset

else:
    with st.spinner("App is loading, please wait..."):
        df = pd.read_csv("output.csv")
        df = preprocess_dataframe(df)

        seed = int(time.time() % (2 ** 16))
        text = df[["full_text"]]
        labels = df[["is_positive"]]
        train_df, val_df, train_labels, val_labels = train_test_split(text, labels, random_state=seed, test_size=0.2,
                                                                      stratify=labels)

        vocab, vocab_size = build_vocab(train_df)
        train_dataset, val_dataset = create_datasets(vocab, train_df, train_labels, val_df, val_labels)

        st.session_state.vocab = vocab
        st.session_state.vocab_size = vocab_size
        st.session_state.train_dataset = train_dataset
        st.session_state.val_dataset = val_dataset


st.title("Cubicle Sentiment Analysis App")

st.write("First, start by picking a model to train.")
model_name = st.selectbox("Choose a model", ("GRU", "LSTM", "RNN"), key="model_name")

st.write("Next, specify some training parameters.")

embedding_dim = st.selectbox("Embedding dimension for text", (32, 64, 128, 256), key="embedding")
hidden_dim = st.selectbox("Hidden dimension for models", (32, 64, 128, 256), key="hidden")
batch_size = st.selectbox("Batch size", (32, 64, 128, 256, 512), key="batch")
learning_rate = st.selectbox("Learning rate", (0.01, 0.005, 1e-3, 5e-4, 1e-4), key="lr")
num_epochs = st.number_input("Number of training epochs", min_value=1, value=5, step=1, key="num_epochs")
early_epochs = st.number_input("Early stopping epochs (stopping based on average batch validation loss)",
                               min_value=1, value=1, step=1, key="early_epochs")

st.write("Now, train the model! If you have already trained a model with the same name, it will be overwritten.")

if st.button("Train model", key="train_model"):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model = Model(model_name, vocab_size, embedding_dim, hidden_dim)

    def callback(epoch, val_loss, val_acc):
        header_text.markdown("Training information:")
        info_df = pd.DataFrame({
            "Info": ["Epoch", "Validation Loss", "Validation Accuracy"],
            "Value": [f"{epoch}", f"{val_loss:.4f}", f"{val_acc:.2f}"]
        })
        status_text.dataframe(info_df,
                              hide_index=True,
                              column_config={
                                  "Info": st.column_config.Column("Info", alignment="left"),
                                  "Value": st.column_config.Column("Value", alignment="left"),
                              },
                              width="stretch"
                              )

    with st.spinner("Model training, please wait...", show_time=True):
        header_text = st.empty()
        status_text = st.empty()
        results, epochs = train_model(model, model_name, device, learning_rate,
                                      num_epochs, early_epochs, train_loader, val_loader, callback)

    st.success(f"Training complete!\n\nEpochs Trained: {epochs}")

    if "trained_models" not in st.session_state:
        st.session_state.trained_models = set()
    st.session_state.trained_models.add(model_name)

    result_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "Specificity", "F1 Score"],
        "Result": [
            f"{results["accuracy"].item():.2f}",
            f"{results["precision"].item():.2f}",
            f"{results["recall"].item():.2f}",
            f"{results["specificity"].item():.2f}",
            f"{results["f1"].item():.2f}"
        ]
    })

    st.write("Metrics for validation data from best performing model:")
    st.dataframe(result_df,
                 hide_index=True,
                 column_config={
                     "Metric": st.column_config.Column("Metric", alignment="left"),
                     "Result": st.column_config.Column("Result", alignment="left"),
                 },
                 width="stretch"
                 )

st.write("After training, input some text, and the model will try to determine the sentiment.")
trained_models = list(st.session_state.get("trained_models", set()))
trained_models.sort(key= lambda x: (x.split(",")[0], int(re.search(r'Embedding Dim: (\d+)', x).group(1)), 
                                                     int(re.search(r'Hidden Dim: (\d+)', x).group(1))))
trained_model = st.selectbox("Model for prediction", trained_models, key="trained_model")
input_text = st.text_input("Type some review text", key="review_text")

if st.button("Predict sentiment", key="predict"):
    if "trained_models" not in st.session_state:
        st.error("No model has been trained yet, sentiment analysis not possible.")
        st.stop()

    if len(input_text.strip()) == 0:
        st.error("Please input some text for model prediction.")
        st.stop()

    model = Model(trained_model, vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    load_checkpoint(f"{trained_model}.pth.tar", model)
    model = model.to(device)
    model.eval()

    cleaned_input = clean_text(input_text)
    if len(cleaned_input.strip()) == 0:
        sequence = torch.zeros(64, dtype=torch.long)
    else:
        sequence = torch.tensor(text_to_sequence(cleaned_input, vocab), dtype=torch.long)

    sequence = sequence.unsqueeze(0).to(device)
    with torch.no_grad():
        output = torch.sigmoid(model(sequence))

    sentiment = ":green[Positive] :+1:" if output.item() > 0.5 else ":red[Negative] :-1:"
    st.markdown(f"Predicted sentiment: {sentiment}")
