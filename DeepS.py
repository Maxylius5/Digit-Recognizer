import pandas as pd
import numpy as np 

# Load data
traindata = pd.read_csv(r'D:\Dataset\traindata.csv')
testdata = pd.read_csv(r'D:\Dataset\Testdata.csv')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    e_x = np.exp(x - np.max(x))  # for numerical stability
    return e_x / np.sum(e_x)

# Initialize weights and biases
input_size = 784
hidden1_size = 20
hidden2_size = 20
output_size = 10

weights1_2 = np.random.randn(input_size, hidden1_size) * 0.1
weights2_3 = np.random.randn(hidden1_size, hidden2_size) * 0.1
weights3_4 = np.random.randn(hidden2_size, output_size) * 0.1

biases1_2 = np.zeros(hidden1_size)
biases2_3 = np.zeros(hidden2_size)
biases3_4 = np.zeros(output_size)

learning_rate = 0.01
epochs = 10
batch_size = 32

# One-hot encoding
def one_hot(y):
    one_hot = np.zeros((y.size, 10))
    one_hot[np.arange(y.size), y] = 1
    return one_hot

# Training loop
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    
    # Shuffle data
    shuffled_data = traindata.sample(frac=1).reset_index(drop=True)
    
    for i in range(0, len(shuffled_data), batch_size):
        batch = shuffled_data[i:i+batch_size]
        
        # Get batch data
        labels = batch.iloc[:, 0].values.astype(int)
        pixels = batch.iloc[:, 1:].values.astype(np.float32) / 255.0
        
        # Forward pass
        # Layer 1 to 2
        z1 = np.dot(pixels, weights1_2) + biases1_2
        a1 = sigmoid(z1)
        
        # Layer 2 to 3
        z2 = np.dot(a1, weights2_3) + biases2_3
        a2 = sigmoid(z2)
        
        # Layer 3 to output
        z3 = np.dot(a2, weights3_4) + biases3_4
        output = softmax(z3)
        
        # One-hot encode labels
        y_true = one_hot(labels)
        
        # Backward pass
        # Output layer error
        error = output - y_true
        delta3 = error  # For softmax + cross-entropy
        
        # Hidden layer 2 error
        error2 = np.dot(delta3, weights3_4.T)
        delta2 = error2 * sigmoid_derivative(a2)
        
        # Hidden layer 1 error
        error1 = np.dot(delta2, weights2_3.T)
        delta1 = error1 * sigmoid_derivative(a1)
        
        # Update weights and biases
        weights3_4 -= learning_rate * np.dot(a2.T, delta3)
        biases3_4 -= learning_rate * np.sum(delta3, axis=0)
        
        weights2_3 -= learning_rate * np.dot(a1.T, delta2)
        biases2_3 -= learning_rate * np.sum(delta2, axis=0)
        
        weights1_2 -= learning_rate * np.dot(pixels.T, delta1)
        biases1_2 -= learning_rate * np.sum(delta1, axis=0)

# Testing
correct = 0
total = len(testdata)

for n in range(total):
    row = testdata.iloc[n]
    label = int(row[0])
    pixels = row[1:].values.astype(np.float32) / 255.0
    
    # Forward pass
    z1 = np.dot(pixels, weights1_2) + biases1_2
    a1 = sigmoid(z1)
    
    z2 = np.dot(a1, weights2_3) + biases2_3
    a2 = sigmoid(z2)
    
    z3 = np.dot(a2, weights3_4) + biases3_4
    output = softmax(z3)
    
    prediction = np.argmax(output)
    
    if prediction == label:
        correct += 1

accuracy = correct * 100 / total
print(f"Accuracy: {accuracy:.2f}%")