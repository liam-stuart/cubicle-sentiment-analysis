import json
import logging
import streamlit as st
import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataset import clean_text, text_to_sequence, ReviewDataset
from model import get_model
from utils import collate_fn, load_checkpoint
from train import train_model


logger = logging.getLogger("cubicle-app")
logging.basicConfig(level=logging.ERROR)
device = "cuda" if torch.cuda.is_available() else "cpu"

st.title("Cubicle Sentiment Analysis App")

st.write("First, start by picking a model to train.")
model_name = st.selectbox("Choose a model", ("GRU", "LSTM", "RNN"))

st.write("Next, specify some training parameters.")

embedding_dim = st.selectbox("Embedding dimension for text", (32, 64, 128, 256))
hidden_dim = st.selectbox("Hidden dimension for models", (32, 64, 128, 256))
batch_size = st.selectbox("Batch size", (32, 64, 128, 256, 512))
learning_rate = st.selectbox("Learning rate", (0.01, 0.005, 1e-3, 5e-4, 1e-4))
num_epochs = st.number_input("Number of training epochs", min_value=1, value=5, step=1)
early_epochs = st.number_input("Early stopping epochs (stopping based on average batch validation loss)",
                               min_value=1, value=1, step=1)

st.write("Now, train the model! If you have already trained a model with the same name, it will be overwritten.")

if st.button("Train model"):
    try:
        with open("vocab.json", "r") as f:
            vocab = json.load(f)
    except FileNotFoundError:
        st.error("No vocabulary found, please run dataset.py to generate a vocabulary.")
        exit()

    train_dataset = ReviewDataset(vocab=vocab, train=True)
    val_dataset = ReviewDataset(vocab=vocab, train=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    vocab_size = max(vocab.values()) + 1
    model = get_model(model_name, vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim)

    def on_epoch_end(epoch, val_loss, val_acc):
        status_text.markdown(
            "TRAINING INFORMATION\n\n"
            "====================\n\n"
            f"Epoch: {epoch}\n\n"
            f"Validation Loss: {val_loss:.4f}\n\n"
            f"Validation Accuracy: {val_acc:.2f}%"
        )

    with st.spinner("Model training, please wait...", show_time=True):
        status_text = st.empty()
        try:
            results, epochs = train_model(model, model_name, device, learning_rate,
                                          num_epochs, early_epochs, train_loader, val_loader, on_epoch_end)
        except Exception as e:
            logger.error(e)
            st.error("Model training failed. See terminal output for further details.")
            exit()

    status_text.empty()
    st.success(f"Training complete!\n\nEpochs Trained: {epochs}")
    if "trained_models" not in st.session_state:
        st.session_state.trained_models = set()
    st.session_state.trained_models.add(model_name)
    result_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "Specificity", "F1 Score"],
        "Result": [
            round(results["accuracy"].item(), 2),
            round(results["precision"].item(), 2),
            round(results["recall"].item(), 2),
            round(results["specificity"].item(), 2),
            round(results["f1"].item(), 2)
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
trained_models = st.session_state.get("trained_models", ())
trained_model = st.selectbox("Model for prediction", trained_models)
input_text = st.text_input("Type some review text")

if st.button("Predict sentiment"):
    if "trained_models" not in st.session_state:
        st.error("No model has been trained yet, sentiment analysis not possible.")
        exit()

    if len(input_text) == 0:
        st.error("Please input some text for model prediction.")
        exit()

    with open("vocab.json", "r") as f:
        vocab = json.load(f)

    vocab_size = max(vocab.values()) + 1
    model = get_model(trained_model, vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    load_checkpoint(f"{trained_model}.pth.tar", model, device)

    cleaned_input = clean_text(input_text)
    if len(cleaned_input) == 0:
        st.error("Text is empty after internal cleaning process, please modify input.")
        exit()

    sequence = torch.tensor(text_to_sequence(cleaned_input, vocab), dtype=torch.long)
    sequence = sequence.unsqueeze(0).to(device)
    output = torch.sigmoid(model(sequence))

    sentiment = ":green[Positive] :+1:" if output > 0.5 else ":red[Negative] :-1:"
    st.markdown(f"Predicted sentiment: {sentiment}")
