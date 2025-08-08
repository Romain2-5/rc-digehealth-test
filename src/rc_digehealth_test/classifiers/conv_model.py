import torch.nn as nn
import torch
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adadelta
from tqdm import tqdm
from torchaudio.transforms import MelSpectrogram
import copy


class BowelSoundCNN(nn.Module):
    def __init__(self, num_classes=2, dropout=0.1, fs=8000):
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
            nn.Linear(64, num_classes)  # Output: (B, 2)
        )

    def forward(self, x):

        x = self.mel_spec(x)
        x = x.unsqueeze(1)

        for layer in self.net:
            x = layer(x)
            # Debug: print intermediate shapes
            # if isinstance(layer, nn.Flatten):
            #     print(f"after Flatten: {x.shape}")
        return x


class SegmentDataset(Dataset):
    def __init__(self, data, labels):
        assert len(data) == len(labels), "Mismatched mel and label lengths"
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]

        # Ensure it's a tensor
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype=torch.long)

        return x, y


def train_model(model, train_dataset, val_dataset=None, epochs=30, batch_size=128, learning_rate=1.0,
                training_sampler=None, early_stop_n_epochs=10, device='cuda' if torch.cuda.is_available() else 'cpu'):

    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=training_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None

    criterion = CrossEntropyLoss()
    optimizer = Adadelta(model.parameters(), lr=learning_rate)

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)              # shape (B, 2)
            loss = criterion(outputs, labels)    # cross-entropy with logits
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / total
        accuracy = correct / total * 100
        print(f"[Epoch {epoch:02d}] Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Validation
        if val_loader:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for val_inputs, val_labels in val_loader:
                    val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                    val_outputs = model(val_inputs)
                    _, val_preds = val_outputs.max(1)
                    val_correct += (val_preds == val_labels).sum().item()
                    val_total += val_labels.size(0)
            val_acc = val_correct / val_total * 100
            print(f"            Validation Accuracy: {val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_model_wts = copy.deepcopy(model.state_dict())
                best_val_acc = val_acc
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if early_stop_n_epochs is not None:
                    if epochs_without_improvement >= early_stop_n_epochs:
                        print(f"Early stopping triggered after {epoch}. Best val accuracy: {best_val_acc:.2f}%")
                        break

    if val_loader:
        model.load_state_dict(best_model_wts)

    return model
