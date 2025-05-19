'''import torch
import pandas as pd
import f'''

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Data Loading and Preparation
def load_data(train_path, test_path):
    # Load CSV files
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Split features and labels
    X_train = train_data.iloc[:, 1:].values.astype(np.float32) / 255.0  # Normalize to [0, 1]
    y_train = train_data.iloc[:, 0].values.astype(np.int64)
    X_test = test_data.iloc[:, 1:].values.astype(np.float32) / 255.0
    y_test = test_data.iloc[:, 0].values.astype(np.int64)
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, device=device)
    y_train = torch.tensor(y_train, device=device)
    X_test = torch.tensor(X_test, device=device)
    y_test = torch.tensor(y_test, device=device)
    
    return X_train, y_train, X_test, y_test

# 2. Define the Neural Network
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 60)
        self.fc3 = nn.Linear(60, 20)
        self.fc4 = nn.Linear(20, 10)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        x = self.fc4(x)  # No activation on output (we'll use CrossEntropyLoss)
        return x

# 3. Training Function
def train_model(model, X_train, y_train, epochs=100, batch_size=32, learning_rate=0.01):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Create DataLoader
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in loader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Print epoch statistics
        epoch_loss = running_loss / len(loader)
        epoch_acc = 100 * correct / total
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

# 4. Evaluation Function
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == y_test).sum().item()
        accuracy = 100 * correct / len(y_test)
        print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

# 5. Save and Load Functions
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(path):
    model = MNISTNet().to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")
    return model

# Main execution
if __name__ == "__main__":
    # Paths to your data D:\AI-digitRecognizer
    train_path = r'D:\AI-digitRecognizer\traindata.csv'
    test_path = r'D:\Dataset\Mostdata.csv'
    model_path = r'D:\Dataset\model_parameters\mnist_model.pth'
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Load data
    X_train, y_train, X_test, y_test = load_data(train_path, test_path)
    
    # Initialize model
    model = MNISTNet().to(device)
    print(model)
    
    # Train model
    train_model(model, X_train, y_train, epochs=50, batch_size=8, learning_rate=0.1)
    
    # Evaluate model
    test_accuracy = evaluate_model(model, X_test, y_test)
    
    # Save model
    save_model(model, model_path)
    
    # To load later:
    # loaded_model = load_model(model_path)
    # evaluate_model(loaded_model, X_test, y_test)