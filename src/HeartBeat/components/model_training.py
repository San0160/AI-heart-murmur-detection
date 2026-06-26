from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch
from collections import Counter
import copy
from HeartBeat.config.configuration import ModelTrainerConfig


# Components
class HeartMurmurLSTM(nn.Module):
    def __init__(self, config: ModelTrainerConfig ):
        super(HeartMurmurLSTM, self).__init__()
        self.config = config
        
        self.lstm = nn.LSTM(
            input_size  = self.config.input_size,
            hidden_size = self.config.hidden_size,  # 64 hidden units
            num_layers  = self.config.num_layers,   # 1 layer
            batch_first = True,
            dropout     = config.dropout if config.num_layers > 1 else 0
        )

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 32),  # 32 instead of 64
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(32, config.num_classes),
            )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.classifier(out)
        return out
    
def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):
        """Convert numpy arrays to PyTorch DataLoaders"""

        def to_tensor(X, y):
            X_t = torch.tensor(X, dtype=torch.float32)
            y_t = torch.tensor(y, dtype=torch.float32)
            return TensorDataset(X_t, y_t)

        train_loader = DataLoader(to_tensor(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(to_tensor(X_val,   y_val),   batch_size=batch_size)
        test_loader  = DataLoader(to_tensor(X_test,  y_test),  batch_size=batch_size)

        return train_loader, val_loader, test_loader

def load_transformed_data(self):
    path = self.config.transformed_data_path

    X_train = np.load(path / "X_train.npy")
    X_val   = np.load(path / "X_val.npy")
    X_test  = np.load(path / "X_test.npy")

    y_train = np.load(path / "y_train.npy")
    y_val   = np.load(path / "y_val.npy")
    y_test  = np.load(path / "y_test.npy")

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, class_weights = None, patience=15):
    """
    model:        HeartMurmurLSTM instance
    train_loader: training DataLoader
    val_loader:   validation DataLoader
    epochs:       number of training epochs
    lr:           learning rate
    class_weights: optional tensor of class weights
    patience:      early stopping patience (epochs without improvement)
    """
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, verbose=True
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # ── Early stopping state ───────────────────────────────────
    best_val_loss  = float('inf')
    best_model     = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        # ── Training ──────────────────────────────────────────
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.argmax(dim=1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            labels = y_batch.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)

        # ── Validation ────────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch.argmax(dim=1))
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                labels = y_batch.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += len(labels)

        # ── Metrics ───────────────────────────────────────────
        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        train_acc = train_correct / train_total * 100
        val_acc = val_correct / val_total * 100

        history["train_loss"].append(train_loss_avg)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss_avg)
        history["val_acc"].append(val_acc)

        scheduler.step(val_loss_avg)

        print(f"Epoch [{epoch+1:02d}/{epochs}] "
            f"Train Loss: {train_loss_avg:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss_avg:.4f} | Val Acc: {val_acc:.2f}% | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # ── Early stopping ────────────────────────────────────
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_model    = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(f"  ✓ Best model saved (val_loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n Early stopping at epoch {epoch+1} — no improvement for {patience} epochs")
                break
    
    # ── Restore best model ────────────────────────────────────
    model.load_state_dict(best_model)
    print(f"\nTraining complete. Best Val Loss: {best_val_loss:.4f}")
    return model, history