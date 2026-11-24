"""Small tensor-based classification metrics used by the final evaluation."""

import torch


def accuracy(y_true, y_pred):
    """Return the fraction of correct predictions, or zero for empty input."""
    y_true = torch.as_tensor(y_true)
    y_pred = torch.as_tensor(y_pred)
    if y_true.numel() == 0:
        return 0.0
    return float((y_true == y_pred).double().mean().item())


def confusion_matrix(y_true, y_pred, num_classes=6):
    """Count true/predicted label pairs; rows are true and columns predicted."""
    y_true = torch.as_tensor(y_true, dtype=torch.long)
    y_pred = torch.as_tensor(y_pred, dtype=torch.long)
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for true_label, predicted_label in zip(y_true.tolist(), y_pred.tolist()):
        matrix[true_label, predicted_label] += 1
    return matrix


def per_class_accuracy(confusion):
    """Divide each diagonal count by its true-class row total."""
    confusion = torch.as_tensor(confusion)
    row_totals = confusion.sum(dim=1)
    correct = confusion.diag()
    return torch.where(
        row_totals > 0,
        correct.float() / row_totals.float(),
        torch.zeros_like(correct, dtype=torch.float32),
    )
