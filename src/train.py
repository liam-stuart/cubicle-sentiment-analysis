import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from utils import save_checkpoint, EarlyStopper


def train_model(model, model_name, device, learning_rate, num_epochs,
                early_epochs, train_loader, val_loader, callback=None):
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler(device=device)
    early_stopper = EarlyStopper(early_epochs)
    for epoch in range(num_epochs):
        loop = tqdm(train_loader)
        model.train()
        num_samples_train = 0
        num_correct_train = 0
        total_train_loss = 0
        for i, (X, y) in enumerate(loop):
            X, y = X.to(device), y.to(device)
            num_samples_train += X.shape[0]
            with torch.amp.autocast(device_type=device):
                outputs = model(X)
                loss = criterion(outputs, y.unsqueeze(1))
            total_train_loss += loss.item()
            outputs = outputs > 0.5
            num_correct_train += (outputs == y.unsqueeze(1)).sum().item()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        total_train_loss /= len(train_loader)
        train_acc = 100 * num_correct_train / num_samples_train

        model.eval()
        num_samples_val = 0
        num_correct_val = 0
        total_val_loss = 0
        with torch.no_grad():
            for (X, y) in val_loader:
                num_samples_val += X.shape[0]
                with torch.amp.autocast(device_type=device):
                    outputs = model(X)
                    loss = criterion(outputs, y.unsqueeze(1))
                total_val_loss += loss.item()
                outputs = outputs > 0.5
                num_correct_val += (outputs == y.unsqueeze(1)).sum().item()

        total_val_loss /= len(val_loader)
        val_acc = 100 * num_correct_val / num_samples_val

        if callback:
            callback(epoch + 1, total_train_loss, train_acc, total_val_loss, val_acc)

        if early_stopper.early_stop(total_val_loss):
            break

    save_checkpoint(model, f"{model_name}.pth.tar")
    return round(train_acc, 2), round(val_acc, 2), epoch + 1
