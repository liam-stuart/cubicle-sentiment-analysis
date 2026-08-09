import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
    BinaryF1Score
)
from utils import EarlyStopper


def train_model(model, model_name, device, learning_rate, num_epochs,
                early_epochs, train_loader, val_loader, callback=None):
    model = model.to(device)
    # Dataset contains many more positive examples, so we weight them lower
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.1]))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler(device=device)
    early_stopper = EarlyStopper(early_epochs)
    for epoch in range(num_epochs):
        loop = tqdm(train_loader)
        model.train()
        for i, (X, y) in enumerate(loop):
            X, y = X.to(device), y.to(device)
            with torch.amp.autocast(device_type=device):
                outputs = model(X)
                outputs = outputs.squeeze(-1)
                loss = criterion(outputs, y)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        val_loss = 0
        metrics = MetricCollection({
            'accuracy': BinaryAccuracy(),
            'precision': BinaryPrecision(),
            'recall': BinaryRecall(),
            'specificity': BinarySpecificity(),
            'f1': BinaryF1Score()
        })
        with torch.no_grad():
            for (X, y) in val_loader:
                with torch.amp.autocast(device_type=device):
                    outputs = model(X)
                    outputs = outputs.squeeze(-1)
                    loss = criterion(outputs, y)

                val_loss += loss.item()
                probs = torch.sigmoid(outputs)
                metrics.update(probs, y)

        val_loss /= len(val_loader)
        results = metrics.compute()

        if callback:
            callback(epoch + 1, val_loss, results["accuracy"].item())

        if early_stopper.early_stop(val_loss, model, model_name, results):
            break

        metrics.reset()

    return early_stopper.results, epoch + 1
