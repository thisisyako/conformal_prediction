import numpy as np
import torch
from torch.amp import autocast, GradScaler
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.classification.models import freeze_backbone

EPOCHS_FC = 8
EPOCHS_FT = 25

LEARN_RATE_FC = 1e-3  # Plus élevé pour la tête linéaire
LEARN_RATE_FT = 6e-5  # Faible pour le fine-tuning

WEIGHT_DECAY_FC = 1e-2
WEIGHT_DECAY_FT = 3e-2


@torch.no_grad()
def evaluate(model, loader, criterion, device, is_cuda):
    model.eval()
    total_loss, total = 0.0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(
            device, non_blocking=True
        )

        with autocast(
            device_type=device.type, enabled=is_cuda
        ):  # Précision mixte sur la validation
            outputs = model(images)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)

        total += labels.size(0)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

    avg_loss = total_loss / total if total else 0.0
    acc = (np.array(all_preds) == np.array(all_labels)).sum() / total if total else 0.0

    # Compute Macro and Weighted F1 Scores
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return avg_loss, acc, f1_macro, f1_weighted, all_preds, all_labels


def train_model(
    model, train_loader, optimizer, scheduler, criterion, epochs, device, is_cuda
):
    scaler = GradScaler(
        device=device.type, enabled=is_cuda
    )  # Pour la précision mixte (FP16)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1_macro": [],
        "val_f1_weighted": [],
    }

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )

            optimizer.zero_grad()

            # Pass Avant + Backward en précision mixte
            with autocast(device_type=device.type, enabled=is_cuda):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)

        scheduler.step()

        train_loss = running_loss / len(train_loader.dataset)

        history["train_loss"].append(train_loss)

        print(f"Epoch {epoch+1:2d}/{epochs} | Train Loss: {train_loss:.4f}")

    return history


def train_classifier(model, train_loader, criterion, device, is_cuda):
    freeze_backbone(model, freeze=True)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARN_RATE_FC,
        weight_decay=WEIGHT_DECAY_FC,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS_FC)

    _ = train_model(
        model,
        train_loader,
        optimizer,
        scheduler,
        criterion,
        epochs=EPOCHS_FC,
        device=device,
        is_cuda=is_cuda,
    )


def train_backbone(model, train_loader, criterion, device, is_cuda):

    freeze_backbone(model, freeze=True)
    for block in model.features[-6:]:
        for p in block.parameters():
            p.requires_grad = True

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARN_RATE_FT,
        weight_decay=WEIGHT_DECAY_FT,
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS_FT)

    _ = train_model(
        model,
        train_loader,
        optimizer,
        scheduler,
        criterion,
        epochs=EPOCHS_FT,
        device=device,
        is_cuda=is_cuda,
    )
