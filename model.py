import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def save_checkpoint(model, optimizer, epoch, file_path="model_checkpoint.pth"):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, file_path)
    print(f"Checkpoint saved at epoch {epoch}")


def load_checkpoint(input_size, hidden_size, num_layers, output_size, learning_rate, file_path="model_checkpoint.pth"):
    checkpoint = torch.load(file_path)
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    model.device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    model.to(model.device)
    print(f"Checkpoint loaded from epoch {epoch}")
    return model, optimizer, epoch


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, 
                scheduler=None, early_stopping=None, grad_clip=None, 
                checkpoint_interval=5, checkpoint_path="model_checkpoint.pth"):
    model.train()
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        epoch_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(model.device), targets.to(model.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(model.device), targets.to(model.device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if scheduler:
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if early_stopping:
                early_stopping.counter = 0
        else:
            if early_stopping:
                early_stopping.counter += 1
                if early_stopping.counter >= early_stopping.patience:
                    print("Early stopping triggered.")
                    break

        # Save every 'checkpoint_interval' epochs
        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, epoch + 1, checkpoint_path)

        model.train()
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_loss:.4f}")
    return train_losses, val_losses


def forecast(model, data_loader):
    model.eval()
    forecasts = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(model.device)
            outputs = model(inputs)
            forecasts.append(outputs.cpu().numpy())
    return np.concatenate(forecasts)


def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(model.device), targets.to(model.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(data_loader)


def plot_training_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Time")
    plt.legend()
    plt.show()


def create_data_loader(X, y, batch_size, shuffle=True):
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
