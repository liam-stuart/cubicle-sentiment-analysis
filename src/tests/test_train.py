import pytest
from unittest.mock import patch
import torch
from torch.utils.data import TensorDataset, DataLoader
from model import Model
from train import train_model


checkpoint_file = "src/GRU_32_32.pth.tar"
model = Model("GRU", 100, 32, 32)
dummy_model_args = ["GRU", 32, 32]
fake_results = {
    "accuracy": torch.tensor(0.9),
    "precision": torch.tensor(0.9),
    "recall": torch.tensor(0.9),
    "specificity": torch.tensor(0.9),
    "f1": torch.tensor(0.9),
}

train_data = torch.randint(0, 50, (50, 50), dtype=torch.long)
train_labels = torch.randint(0, 1, (50, 1), dtype=torch.float32)
val_data = torch.randint(0, 50, (10, 50), dtype=torch.long)
val_labels = torch.randint(0, 1, (10, 1), dtype=torch.float32)

train_dataset = TensorDataset(train_data, train_labels)
val_dataset = TensorDataset(val_data, val_labels)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


def callback(*args):
    pass


def test_train_model_runs(remove_tar):
    _, epochs = train_model(model, dummy_model_args, "cpu", 0.01, 3, 4, train_loader, val_loader, callback, False)
    assert epochs == 3


@patch("train.EarlyStopper")
def test_train_model_stops_early(mock_early_stopper):
    mock_instance = mock_early_stopper
    mock_instance.results = fake_results
    mock_instance.early_stop.return_value = True
    results, epochs = train_model(model, dummy_model_args, "cpu", 0.01, 3, 4, train_loader, val_loader, callback, False)
    for result in results:
        assert results[result] == fake_results[result]
    assert epochs == 1


def test_train_model_stops_fails_no_file():
    with pytest.raises(FileNotFoundError):
        assert train_model(model, dummy_model_args, "cpu", 0.01, 3, 4,
                           train_loader, val_loader, callback, True)
