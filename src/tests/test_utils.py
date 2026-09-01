import os
import torch
import torch.nn as nn
from utils import EarlyStopper, load_checkpoint, save_checkpoint
from model import Model


dummy_model = Model("GRU", 100, 32, 32)
dummy_model_args = ["GRU", 32, 32]
dummy_lr = 0.01
dummy_optimizer = torch.optim.Adam(dummy_model.parameters(), lr=dummy_lr)
dummy_optimizer.step()
checkpoint_file = "models/GRU_32_32.pth.tar"
fake_results = {
    "accuracy": torch.tensor(0.9),
    "precision": torch.tensor(0.9),
    "recall": torch.tensor(0.9),
    "specificity": torch.tensor(0.9),
    "f1": torch.tensor(0.9),
}


def test_early_stopper_init():
    early_stopper = EarlyStopper()
    assert early_stopper.patience == 3
    assert early_stopper.counter == 0
    assert early_stopper.min_validation_loss == float('inf')


def test_early_stopper_lower_min(remove_tar):
    early_stopper = EarlyStopper()
    early_stopper.min_validation_loss = 10
    assert not early_stopper.early_stop(5, dummy_model, dummy_model_args, dummy_optimizer, fake_results)
    assert os.path.exists(checkpoint_file)


def test_early_stopper_counter_increments():
    early_stopper = EarlyStopper()
    early_stopper.min_validation_loss = 10
    assert not early_stopper.early_stop(20, dummy_model, dummy_model_args, dummy_optimizer, fake_results)
    assert not os.path.exists(checkpoint_file)
    assert early_stopper.counter == 1


def test_early_stopper_patience_exceeded():
    early_stopper = EarlyStopper()
    early_stopper.min_validation_loss = 10
    early_stopper.counter = 2
    assert early_stopper.early_stop(20, dummy_model, dummy_model_args, dummy_optimizer, fake_results)
    assert not os.path.exists(checkpoint_file)


def test_save_checkpoint(remove_tar):
    save_checkpoint(checkpoint_file, dummy_model, dummy_optimizer)
    assert os.path.exists(checkpoint_file)


def test_load_checkpoint(remove_tar):
    save_checkpoint(checkpoint_file, dummy_model, dummy_optimizer)
    dummy_model_new = Model("GRU", 100, 32, 32)
    for param in dummy_model_new.parameters():
        param.data = nn.parameter.Parameter(torch.rand_like(param))
    dummy_optimizer_new = torch.optim.Adam(dummy_model_new.parameters(), lr=dummy_lr * 2)
    load_checkpoint(checkpoint_file, dummy_model_new, dummy_optimizer_new, dummy_lr * 3)
    old_model_sd = dummy_model.state_dict()
    new_model_sd = dummy_model_new.state_dict()
    old_optimizer_sd = dummy_optimizer.state_dict()
    new_optimizer_sd = dummy_optimizer_new.state_dict()

    assert len(old_model_sd) == len(new_model_sd)
    assert len(old_optimizer_sd['param_groups']) == len(new_optimizer_sd['param_groups'])

    for weights in old_model_sd:
        assert torch.equal(old_model_sd[weights], new_model_sd[weights])

    for group_old, group_new in zip(old_optimizer_sd['param_groups'], new_optimizer_sd['param_groups']):
        for key in group_old:
            if key not in ['lr', 'params']:
                assert group_old[key] == group_new[key]
            elif key == 'lr':
                assert group_new[key] == dummy_lr * 3

    for state_id in old_optimizer_sd["state"]:
        state_old = old_optimizer_sd["state"][state_id]
        state_new = new_optimizer_sd["state"][state_id]

        for key in state_old:
            val_old = state_old[key]
            val_new = state_new[key]

            if isinstance(val_old, torch.Tensor):
                assert torch.equal(val_old, val_new)
            else:
                assert val_old == val_new
