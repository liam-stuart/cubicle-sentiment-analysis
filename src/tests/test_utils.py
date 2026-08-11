import os
import pytest
import torch
from utils import EarlyStopper, load_checkpoint, save_checkpoint
from model import Model


dummy_model = Model("GRU", 100, 32, 32)
checkpoint_file = "src/GRU.pth.tar"
fake_results = {
    "accuracy": torch.tensor(0.9),
    "precision": torch.tensor(0.9),
    "recall": torch.tensor(0.9),
    "specificity": torch.tensor(0.9),
    "f1": torch.tensor(0.9),
}


@pytest.fixture
def remove_tar():
    yield
    os.remove(checkpoint_file)


def test_early_stopper_init():
    early_stopper = EarlyStopper()
    assert early_stopper.patience == 3
    assert early_stopper.counter == 0
    assert early_stopper.min_validation_loss == float('inf')


def test_early_stopper_lower_min(remove_tar):
    early_stopper = EarlyStopper()
    early_stopper.min_validation_loss = 10
    assert not early_stopper.early_stop(5, dummy_model, "GRU", fake_results)
    assert os.path.exists(checkpoint_file)


def test_early_stopper_counter_increments():
    early_stopper = EarlyStopper()
    early_stopper.min_validation_loss = 10
    assert not early_stopper.early_stop(20, dummy_model, "GRU", fake_results)
    assert not os.path.exists(checkpoint_file)
    assert early_stopper.counter == 1


def test_early_stopper_patience_exceeded():
    early_stopper = EarlyStopper()
    early_stopper.min_validation_loss = 10
    early_stopper.counter = 2
    assert early_stopper.early_stop(20, dummy_model, "GRU", fake_results)
    assert not os.path.exists(checkpoint_file)


def test_save_checkpoint(remove_tar):
    save_checkpoint(dummy_model, checkpoint_file)
    assert os.path.exists(checkpoint_file)


def test_load_checkpoint(remove_tar):
    save_checkpoint(dummy_model, checkpoint_file)
    dummy_model_new = Model("GRU", 100, 32, 32)
    load_checkpoint(checkpoint_file, dummy_model_new)
    old_state_dict = dummy_model.state_dict()
    new_state_dict = dummy_model_new.state_dict()
    for weights in old_state_dict:
        assert old_state_dict[weights].all() == new_state_dict[weights].all()
