import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from utils import save_checkpoint, EarlyStopper


def train(model, device, learning_rate, num_epochs, early_epochs, train_loader, val_loader):
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loop = tqdm(train_loader)
    scaler = torch.amp.GradScaler(device=device)
    early_stopper = EarlyStopper()
    for epoch in range(num_epochs):
        model.train()
        for i, (X, y) in enumerate(loop):
            X, y = X.to(device), y.to(device)
            with torch.amp.autocast(device_type=device):
                outputs = model(X)
                loss = criterion(outputs, y.unsqueeze(1))

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        num_samples = 0
        num_correct = 0
        total_loss = 0
        with torch.no_grad():
            for (X, y) in val_loader:
                num_samples += X.shape[0]
                with torch.amp.autocast(device_type=device):
                    outputs = model(X)
                    loss = criterion(outputs, y.unsqueeze(1))
                total_loss += loss.item()
                outputs = outputs > 0.5
                num_correct += (outputs == y.unsqueeze(1)).sum()

        total_loss /= len(val_loader)
        print(f"EPOCH: {epoch + 1}, LOSS: {total_loss}")
        print(f"Accuracy: {num_correct} / {num_samples}")

        if early_stopper.early_stop(total_loss):
            print(f"No improvement after {early_epochs} epochs, ")
            break

    save_checkpoint(model, optimizer, "checkpoint.pth.tar")
