import pandas as pd
import os
import numpy as np 

traindata = pd.read_csv(r'D:\Dataset\Mostdata.csv')


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x))  # for numerical stability
    return e_x / np.sum(e_x)

def weighted_sum(a,w,b):
    return np.dot(w,a)+b

def compute_activations1_2(row, weights, biases):
    
    pixels = row[1:785].values.astype(np.float32) / 255.0  # Normalize pixel values
    z = np.dot(pixels, weights) + biases  # Weighted sum
    activations = sigmoid(z)
    return pixels, activations



def compute_activations2_3(previous, weights2, biases2):
    m=np.dot(previous, weights2)+biases2
    nextactivations= sigmoid(m)
    return nextactivations



def again(a_current, a_prev, w, b, true_label):
    dc_da=2*(a_current - true_label)
    sigmoid_derivative= a_current * (1-a_current)
    delta= dc_da * sigmoid_derivative
    dw = np.outer(a_prev, delta)
    db = delta

    w -= 0.1 * dw
    b -= 0.1 * db

    return delta, w, b
    

weights1_2 = np.random.randn(784, 20) * 0.1
weights2_3 = np.random.randn(20, 20) * 0.1
weights3_4 = np.random.randn(20, 10) * 0.1

biases1_2 = np.zeros((20,))
biases2_3 = np.zeros((20,))
biases3_4 = np.zeros((10,))

activation2 = np.zeros((20,))
activation3 = np.zeros((20,))
result = np.zeros((10,))
cost = np.zeros((10,))

answer=np.diag(np.full(10,1))

label = 0
image = []
lala = []
average_cost=0
correct=0
leanring_rate=0.01

ep = 100
for epoch in range(ep):
    print(f"Epoch {epoch+1}/{ep}")
    for n in range(69998):

        image, activation2 = compute_activations1_2(traindata.iloc[n], weights1_2, biases1_2)
        activation3 = compute_activations2_3(activation2, weights2_3, biases2_3)

        result = compute_activations2_3(activation3, weights3_4, biases3_4)
        lala=traindata.iloc[n]
        label = int(lala[0])  
        cost = (result - answer[label]) **2
        average_cost+=cost
        delta4, weights3_4, biases3_4 = again(result, activation3, weights3_4, biases3_4, answer[label])
    
        delta3 = np.dot(delta4, weights3_4.T) * (activation3 * (1 - activation3))
        delta2 = np.dot(delta3, weights2_3.T) * (activation2 * (1 - activation2))
        gradient2_3=np.outer(activation2, delta3)
        gradient1_2=np.outer(image, delta2)
        weights2_3 -= leanring_rate * gradient2_3
        biases2_3 -= leanring_rate * delta3
        weights1_2 -= leanring_rate * gradient1_2
        biases1_2 -= leanring_rate * delta2

print("Training succesfully complited")




testdata = pd.read_csv(r'D:\Dataset\Testdata.csv')

for n in range(9998):

    image, activation2 = compute_activations1_2(testdata.iloc[n], weights1_2, biases1_2)
    activation3 = compute_activations2_3(activation2, weights2_3, biases2_3)
    result = compute_activations2_3(activation3, weights3_4, biases3_4)
    row = testdata.iloc[n]
    label = int(row[0])
    result=softmax(result)
    prediction = np.argmax(result)

    if prediction == label:
        correct += 1
print("The accuraccy is", correct*100/9998)


save_dir = r'D:\Dataset\model_parameters'
os.makedirs(save_dir, exist_ok=True)

# Function to save 2D arrays (weights)
def save_weights_to_csv(array, filename):
    pd.DataFrame(array).to_csv(os.path.join(save_dir, filename), index=False, header=False)

# Function to save 1D arrays (biases)
def save_biases_to_csv(array, filename):
    pd.DataFrame(array.reshape(1, -1)).to_csv(os.path.join(save_dir, filename), index=False, header=False)

# Save all parameters
save_weights_to_csv(weights1_2, 'weights1_2.csv')
save_weights_to_csv(weights2_3, 'weights2_3.csv')
save_weights_to_csv(weights3_4, 'weights3_4.csv')

save_biases_to_csv(biases1_2, 'biases1_2.csv')
save_biases_to_csv(biases2_3, 'biases2_3.csv')
save_biases_to_csv(biases3_4, 'biases3_4.csv')

print("All weights and biases saved successfully!")


'''

import pandas as pd

import numpy as np

traindata = pd.read_csv(r'D:\Dataset\traindata.csv')


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x))  # for numerical stability
    return e_x / np.sum(e_x)

def weighted_sum(a,w,b):
    return np.dot(w,a)+b

def compute_activations2(row, weights, biases):
    
    pixels = row[1:785].values.astype(np.float32) / 255.0  # Normalize pixel values
    z = np.dot(pixels, weights) + biases  # Weighted sum
    activations = sigmoid(z)
    return pixels, activations



def compute_activations3(previous, weights2, biases2):
    m=np.dot(previous, weights2)+biases2
    nextactivations= sigmoid(m)
    return nextactivations



def again(a_current, a_prev, w, b, true_label):
    dc_da=2*(a_current - true_label)
    sigmoid_derivative= a_current * (1-a_current)
    delta= dc_da * sigmoid_derivative
    dw = np.outer(a_prev, delta)
    db = delta

    w -= 0.1 * dw
    b -= 0.1 * db

    return delta, w, b
    

weights1_2 = np.random.randn(784, 20) * 0.1
weights2_3 = np.random.randn(20, 20) * 0.1
weights3_4 = np.random.randn(20, 10) * 0.1
biases1_2 = np.zeros((20,))
biases2_3 = np.zeros((20,))
biases3_4 = np.zeros((10,))
activation2 = np.zeros((20,))
activation3 = np.zeros((20,))
result = np.zeros((10,))
cost = np.zeros((10,))

answer=np.diag(np.full(10,1))

label = 0
image = []
lala = []
average_cost=0
correct=0
leanring_rate=0.01
ep = 5
for epoch in range(ep):
    print(f"Epoch {epoch+1}/{ep}")
    for n in range(59995):

        image, activation2 = compute_activations2(traindata.iloc[n], weights1_2, biases1_2)
        activation3 = compute_activations3(activation2, weights2_3, biases2_3)

        result = compute_activations3(activation3, weights3_4, biases3_4)
        lala=traindata.iloc[n]
        label = int(lala[0])  
        cost = (result - answer[label]) **2
        average_cost+=cost
        delta4, weights3_4, biases3_4 = again(result, activation3, weights3_4, biases3_4, answer[label])
    
        delta3 = np.dot(delta4, weights3_4.T) * (activation3 * (1 - activation3))
        delta2 = np.dot(delta3, weights2_3.T) * (activation2 * (1 - activation2))
        gradient2_3=np.outer(activation2, delta3)
        gradient1_2=np.outer(image, delta2)
        weights2_3 -= leanring_rate * gradient2_3
        biases2_3 -= leanring_rate * delta3
        weights1_2 -= leanring_rate * gradient1_2
        biases1_2 -= leanring_rate * delta2

print("Training succesfully complited")




testdata = pd.read_csv(r'D:\Dataset\Testdata.csv')

for n in range(9995):

    image, activation2 = compute_activations2(testdata.iloc[n], weights1_2, biases1_2)
    activation3 = compute_activations3(activation2, weights2_3, biases2_3)
    result = compute_activations3(activation3, weights3_4, biases3_4)
    row = testdata.iloc[n]
    label = int(row[0])
    result=softmax(result)
    prediction = np.argmax(result)

    if prediction == label:
        correct += 1
print("The accuraccy is", correct*100/9995)

'''