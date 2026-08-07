import json
import streamlit as st
import torch
from torch.utils.data import DataLoader
from dataset import clean_text, text_to_sequence, ReviewDataset
from model import get_model
from utils import collate_fn, load_checkpoint
from train import train_model


st.title("Cubicle Sentiment Analysis App")

st.write("First, start by picking a model to train.")
model_name = st.selectbox("Choose a model", ("LSTM"))

st.write("Next, specify some training hyperparameters.")

batch_size = st.selectbox("Batch size", (32, 64, 128, 256, 512))
learning_rate = st.selectbox("Learning rate", (0.01, 0.005, 1e-3, 5e-4, 1e-4))
num_epochs = st.number_input("Number of training epochs", min_value=1, value=5, step=1)
early_epochs = st.number_input("Early stopping epochs (stopping based on average batch validation loss)",
                               min_value=1, value=1, step=1)
device = "cuda" if torch.cuda.is_available() else "cpu"

st.write("Now, train the model!")

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

    model = get_model(model_name, vocab_size=max(vocab.values()) + 1)
    st.session_state.model_name = model_name

    def on_epoch_end(epoch, train_loss, train_acc, val_loss, val_acc):
        status_text.markdown(
            "TRAINING INFORMATION\n\n"
            "====================\n\n"
            f"Epoch: {epoch}\n\n"
            f"Training Loss: {train_loss:.4f}\n\n"
            f"Training Accuracy: {train_acc:.2f}%\n\n"
            f"Validation Loss: {val_loss:.4f}\n\n"
            f"Validation Accuracy: {val_acc:.2f}%"
        )

    with st.spinner("Model training, please wait...", show_time=True):
        status_text = st.empty()
        train_acc, val_acc, epochs = train_model(model, model_name, device, learning_rate,
                                                 num_epochs, early_epochs, train_loader, val_loader, on_epoch_end)

    status_text.empty()
    st.success(f"Training complete!\n\n"
               f"Epochs Trained: {epochs}\n\n"
               f"Final Training Accuracy: {train_acc}%\n\n"
               f"Final Validation Accuracy: {val_acc}%")

st.write("After training, input some text, and the model will try to determine the sentiment.")
input_text = st.text_input("Type some review text")

if st.button("Predict sentiment"):
    if "model_name" not in st.session_state:
        st.error("No model has been trained yet, sentiment analysis not possible.")
        exit()
    model_name = st.session_state.model_name

    with open("vocab.json", "r") as f:
        vocab = json.load(f)

    model = get_model(model_name, vocab_size=max(vocab.values()) + 1)
    load_checkpoint(f"{model_name}.pth.tar", model, device)

    cleaned_input = clean_text(input_text)
    sequence = torch.tensor(text_to_sequence(cleaned_input, vocab), dtype=torch.long)
    sequence = sequence.unsqueeze(0).to(device)
    output = torch.sigmoid(model(sequence))

    sentiment = ":green[Positive]" if output > 0.5 else ":red[Negative]"
    st.markdown(f"Predicted sentiment: {sentiment}")
