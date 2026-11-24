"""Train HARNet and persist the checkpoint with the best validation loss."""

import os
import time
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from prepare import prepare
from model import HARNet


def evaluate(model, X, y, batch_size=64):
    """Compute mean cross-entropy and accuracy without gradient tracking."""
    model.eval()
    loss_sum = 0.0
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for xb, yb in DataLoader(TensorDataset(X, y), batch_size=batch_size):
            logits, _ = model(xb)
            loss_sum += criterion(logits, yb).item()
            correct += (logits.argmax(dim=1) == yb).sum().item()
    return loss_sum / len(y), correct / len(y)


def train():
    """Run the specified 30-epoch Adam training and save model/history files."""
    os.makedirs("results", exist_ok=True)
    tensors = prepare()
    X_train, y_train = tensors["X_train"], tensors["y_train"]
    X_val, y_val = tensors["X_val"], tensors["y_val"]

    # Fix initialization and data-loader randomness for a repeatable run.
    torch.manual_seed(0)
    model = HARNet()
    initial_logits, _ = model(X_train[:64])
    initial_loss = nn.CrossEntropyLoss()(initial_logits, y_train[:64]).item()
    print(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    print(f"Initial loss: {initial_loss:.6f}")

    criterion = nn.CrossEntropyLoss()
    # Shuffle only the training batches; validation remains deterministic.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=64, shuffle=True
    )
    history = []
    best_loss = float("inf")
    best_epoch = 0
    best_train_loss = 0.0
    best_val_accuracy = 0.0
    started = time.time()

    # Train for the required 30 epochs and select by validation loss.
    for epoch in range(1, 31):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            # Limit exploding updates while preserving the gradient direction.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += loss.item() * len(yb)
            train_count += len(yb)

        train_loss = train_loss_sum / train_count
        val_loss, val_accuracy = evaluate(model, X_val, y_val)
        history.append((epoch, train_loss, val_loss, val_accuracy))
        print(
            f"Epoch {epoch:02d}: train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} val_acc={val_accuracy:.4f}"
        )

        # Keep the checkpoint from the best generalization estimate so far.
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_train_loss = train_loss
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "results/best_har_model.pt")

    elapsed = time.time() - started
    torch.save(
        {
            "history": history,
            "initial_loss": initial_loss,
            "best_epoch": best_epoch,
            "best_train_loss": best_train_loss,
            "best_val_loss": best_loss,
            "best_val_accuracy": best_val_accuracy,
            "elapsed_seconds": elapsed,
            "hardware": "CPU",
        },
        "results/training_history.pt",
    )
    return model


if __name__ == "__main__":
    # Running this module directly starts the complete training stage.
    train()
