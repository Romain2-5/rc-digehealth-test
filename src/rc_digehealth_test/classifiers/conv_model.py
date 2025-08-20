import torch.nn as nn
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Sampler, Dataset
from torch.nn import CrossEntropyLoss
from torch.optim import Adadelta, Adam
from tqdm import tqdm
from torchaudio.transforms import MelSpectrogram
import copy
from typing import Optional, Tuple
import numpy as np


class BowelSoundCNN(nn.Module):
    """Convolutional Neural Network for bowel sound classification.

    This model first converts the raw waveform into a mel spectrogram
    and then applies convolutional layers followed by fully connected layers
    for classification.

    Args:
        num_classes (int): Number of output classes.
        dropout (float): Dropout probability.
        fs (int): Sampling rate of input audio.

    Input shape:
        (batch_size, signal_length)

    Output shape:
        (batch_size, num_classes)
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.2, fs: int = 8000) -> None:
        super().__init__()

        self.mel_spec = MelSpectrogram(sample_rate=fs, n_fft=int(fs * 0.05), n_mels=128, hop_length=int(fs * 0.005),
                                  normalized=True,
                                  center=False)

        self.net = nn.Sequential(

            # Conv Block 1
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 1), padding=1),  # (B, 16, 128, 3)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),  # (B, 16, 64, 3)

            # Conv Block 2
            nn.Conv2d(16, 32, kernel_size=(3, 1), padding=1),  # (B, 32, 64, 3)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),  # (B, 32, 32, 3)

            # Conv Block 3
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),  # (B, 64, 32, 3)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),  # (B, 64, 16, 3)

            nn.Dropout(dropout),
            nn.Flatten(),  # (B, 64 * 16 * 3 = 3072)
            nn.Linear(7168, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)  # Output: (B, num_classes)
        )

    def forward(self, x):
        """Forward pass.
        Args:
            x (torch.Tensor): Input waveform tensor of shape (batch_size, signal_length).
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """

        x = self.mel_spec(x)
        x = x.unsqueeze(1)  # Add channel dimension

        for layer in self.net:
            x = layer(x)
            # Debug: print intermediate shapes
            # if isinstance(layer, nn.Flatten):
            #     print(f"after Flatten: {x.shape}")

        return x


class SegmentDataset(Dataset):
    """Dataset for segmented audio data.

    Args:
        data (torch.Tensor or np.ndarray): Input audio segments.
        labels (torch.Tensor or np.ndarray): Corresponding class labels.
    """

    def __init__(self, data, labels) -> None:
        assert len(data) == len(labels), "Mismatched mel and label lengths"
        self.data = data
        self.labels = labels

    def __len__(self) -> int:
        """Return number of samples in the dataset."""

        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample and its label.

        Args:
            index (int): Index of the sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - x: Audio tensor of shape (signal_length,)
                - y: Label tensor (scalar)
        """

        x = self.data[index]
        y = self.labels[index]

        # Ensure it's a tensor
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype=torch.long)

        return x, y



def train_model(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    epochs: int = 30,
    batch_size: int = 128,
    learning_rate: float = 1.0,
    training_sampler: Optional[Sampler] = None,
    early_stop_n_epochs: Optional[int] = 10,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    plot_training: bool = True,
) -> nn.Module:
    """Train a CNN model on bowel sound data.

    Args:
        model (nn.Module): Model to train.
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset, optional): Validation dataset. Defaults to None.
        epochs (int): Number of training epochs. Defaults to 30.
        batch_size (int): Batch size for training. Defaults to 128.
        learning_rate (float): Learning rate. Defaults to 1.0.
        training_sampler (Sampler, optional): Sampler for weighted/balanced training. Defaults to None.
        early_stop_n_epochs (int, optional): Number of epochs without improvement before early stopping. Defaults to 10.
        device (str): Device to use ('cuda' or 'cpu').
        plot_training (bool, optional): Whether to plot training progress. Defaults to True.

    Returns:
        nn.Module: Trained model with best weights (based on validation loss).
    """

    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=training_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None

    criterion = CrossEntropyLoss()
    optimizer = Adadelta(model.parameters(), lr=learning_rate)
    # optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        # --------------------
        # Training
        # --------------------
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)              # shape (B, num_classes)
            loss = criterion(outputs, labels)    # cross-entropy
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / total
        train_losses.append(avg_loss)
        accuracy = correct / total * 100
        print(f"[Epoch {epoch:02d}] Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # --------------------
        # Validation
        # --------------------
        if val_loader:
            model.eval()
            val_running_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for val_inputs, val_labels in val_loader:
                    val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                    val_outputs = model(val_inputs)
                    val_loss = criterion(val_outputs, val_labels)

                    val_running_loss += val_loss.item() * val_inputs.size(0)
                    _, val_preds = val_outputs.max(1)
                    val_correct += (val_preds == val_labels).sum().item()
                    val_total += val_labels.size(0)

            val_avg_loss = val_running_loss / val_total
            val_losses.append(val_avg_loss)

            val_acc = val_correct / val_total * 100
            print(f"Validation Loss: {val_avg_loss:.4f}, Accuracy: {val_acc:.2f}%")

            # --------------------
            # Model selection on val loss
            # --------------------
            if val_avg_loss < best_val_loss:
                best_model_wts = copy.deepcopy(model.state_dict())
                best_val_loss = val_avg_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if early_stop_n_epochs is not None:
                    if epochs_without_improvement >= early_stop_n_epochs:
                        print(f"Early stopping triggered after {epoch}. Best val loss: {best_val_loss:.4f}")
                        break

    # Restore best weights
    model.load_state_dict(best_model_wts)

    if plot_training:
        plt.plot(train_losses, label='Train')
        if val_loader:
            plt.plot(val_losses, label='Val')
            plt.axvline(np.argmin(val_losses), color='r', linestyle='--', label='Best val loss')

        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()

    return model
