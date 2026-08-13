import torch
import torch.nn as nn


class EarlyStopper:
    def __init__(self, patience: int = 3):
        self.patience = patience
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss: float, model: nn.Module, model_args: list,
                   optimizer: torch.optim.Optimizer, results: dict[str, torch.Tensor]) -> bool:
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.results = results
            model_name, embedding_dim, hidden_dim = model_args
            save_checkpoint(f"src/{model_name}_{embedding_dim}_{hidden_dim}.pth.tar", model, optimizer)
        elif validation_loss > (self.min_validation_loss):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def save_checkpoint(filename: str, model: nn.Module, optimizer: torch.optim.Optimizer):
    checkpoint = {"model_state_dict": model.state_dict(),
                  "optimizer_state_dict": optimizer.state_dict()}
    torch.save(checkpoint, filename)


def load_checkpoint(filename: str, model: nn.Module, optimizer: torch.optim.Optimizer | None = None,
                    learning_rate: float | None = None):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
