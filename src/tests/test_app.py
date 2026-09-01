import os
from unittest.mock import patch
import pandas as pd
import torch
from streamlit.testing.v1 import AppTest


fake_data = pd.read_csv("src/tests/test.csv")
zero_calls = 0
original_zeros = torch.zeros


def fake_train_model(*args, **kwargs):
    fake_results = {
        "accuracy": torch.tensor(0.9),
        "precision": torch.tensor(0.9),
        "recall": torch.tensor(0.9),
        "specificity": torch.tensor(0.9),
        "f1": torch.tensor(0.9),
    }
    model_name, embedding_dim, hidden_dim = args[1]
    callback = args[-2]
    callback(1, 0.012345, 0.90)
    file_path = f"models/{model_name}_{embedding_dim}_{hidden_dim}.pth.tar"
    with open(file_path, "w") as f:
        f.write("Super cool model weights.")
    return fake_results, 1


def fake_zeros(*args, **kwargs):
    global zero_calls
    if args == (64,) and kwargs.get("dtype") == torch.long:
        zero_calls += 1

    return original_zeros(*args, **kwargs)


def test_init():
    with patch("app.pd.read_csv", return_value=fake_data):
        with open("models/test.tar", "w") as f:
            f.write("Dummy checkpoint file.")
        at = AppTest.from_file("../app.py", default_timeout=60).run()
        assert not at.exception
        for state in ["vocab", "vocab_size", "train_dataset", "val_dataset"]:
            assert state in at.session_state
        assert not os.path.exists("models/test.tar")


def test_options_work():
    with patch("app.pd.read_csv", return_value=fake_data):
        at = AppTest.from_file("../app.py", default_timeout=60).run()
        at.selectbox("model_name").select("LSTM").run()
        at.selectbox("embedding").select(128).run()
        at.selectbox("hidden").select(64).run()
        at.selectbox("batch").select(512).run()
        at.selectbox("lr").select(5e-4).run()
        at.number_input("num_epochs").decrement().run()
        at.number_input("early_epochs").increment().increment().run()

        selection_keys = ["model_name", "embedding", "hidden", "batch", "lr"]
        expected_selections = ["LSTM", 128, 64, 512, 5e-4]
        input_keys = ["num_epochs", "early_epochs"]
        expected_inputs = [4, 3]

        for selection_key, expected_selection in zip(selection_keys, expected_selections):
            assert at.selectbox(selection_key).value == expected_selection

        for input_key, expected_input in zip(input_keys, expected_inputs):
            assert at.number_input(input_key).value == expected_input


def test_train_model_works(remove_tar):
    with patch("app.pd.read_csv", return_value=fake_data), patch("train.train_model", side_effect=fake_train_model):
        at = AppTest.from_file("../app.py", default_timeout=60).run()
        at.button("train_model").click().run()
        assert os.path.exists("models/GRU_32_32.pth.tar")
        assert at.success[0].value == "Training complete!\n\nEpochs Trained: 1"
        assert at.session_state.trained_models == set(["GRU, Embedding Dim: 32, Hidden Dim: 32"])

        markdown_texts = [m.value for m in at.markdown]
        assert any("Training information:" in text for text in markdown_texts)

        expected_callback = pd.DataFrame({
            "Info": ["Epoch", "Validation Loss", "Validation Accuracy"],
            "Value": ["1", "0.0123", "0.90"]
        })
        assert at.dataframe[0].value.equals(expected_callback)

        expected_final = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "Specificity", "F1 Score"],
            "Result": ["0.90", "0.90", "0.90", "0.90", "0.90"]
        })
        assert at.dataframe[1].value.equals(expected_final)


def test_load_existing_model(remove_tar):
    with patch("app.pd.read_csv", return_value=fake_data), patch("train.train_model", side_effect=fake_train_model):
        at = AppTest.from_file("../app.py", default_timeout=60).run()
        at.button("train_model").click().run()
        at.checkbox("load_model").check().run()
        at.button("train_model").click().run()
        assert len(at.warning) == 0


def test_load_existing_model_fails_no_model():
    with patch("app.pd.read_csv", return_value=fake_data), patch("train.train_model", side_effect=FileNotFoundError):
        at = AppTest.from_file("../app.py", default_timeout=60).run()
        at.checkbox("load_model").check().run()
        at.button("train_model").click().run()
        assert at.warning[0].value == "No checkpoint found for the specified model parameters. Training from scratch."


def test_predict_sentiment_runs():
    with patch("app.pd.read_csv", return_value=fake_data), patch("utils.load_checkpoint", side_effect=None):
        at = AppTest.from_file("../app.py", default_timeout=60)
        at.session_state.trained_models = {"GRU, Embedding Dim: 32, Hidden Dim: 32"}
        at.run()
        at.selectbox("trained_model").select("GRU, Embedding Dim: 32, Hidden Dim: 32").run()
        at.text_input("review_text").input("Super cool test string.").run()
        at.button("predict").click().run()
        assert "Predicted sentiment" in at.markdown[-1].value


def test_predict_sentiment_runs_empty_clean():
    with patch("app.pd.read_csv", return_value=fake_data), patch("utils.load_checkpoint", side_effect=None), \
            patch("app.torch.zeros", side_effect=fake_zeros):
        at = AppTest.from_file("../app.py", default_timeout=60)
        at.session_state.trained_models = {"GRU, Embedding Dim: 32, Hidden Dim: 32"}
        at.run()
        at.selectbox("trained_model").select("GRU, Embedding Dim: 32, Hidden Dim: 32").run()
        at.text_input("review_text").input("a").run()

        at.button("predict").click().run()
        assert zero_calls == 1
        assert "Predicted sentiment" in at.markdown[-1].value


def test_predict_sentiment_fails_no_models():
    with patch("app.pd.read_csv", return_value=fake_data):
        at = AppTest.from_file("../app.py", default_timeout=60).run()
        at.button("predict").click().run()
        assert "No model has been trained yet" in at.error[0].value


def test_predict_sentiment_fails_bad_input():
    with patch("app.pd.read_csv", return_value=fake_data):
        at = AppTest.from_file("../app.py", default_timeout=60)
        at.session_state.trained_models = {"GRU, Embedding Dim: 32, Hidden Dim: 32"}
        at.run()
        at.selectbox("trained_model").select("GRU, Embedding Dim: 32, Hidden Dim: 32").run()
        at.text_input("review_text").input("").run()
        at.button("predict").click().run()
        assert "Please input some text" in at.error[0].value
