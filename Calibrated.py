import pandas as pd
import os
import numpy as np 



save_dir = r'D:\Dataset\model_parameters'
def load_weights_from_csv(filename):
    return pd.read_csv(os.path.join(save_dir, filename), header=None).values

def load_biases_from_csv(filename):
    return pd.read_csv(os.path.join(save_dir, filename), header=None).values.flatten()

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


cost = np.zeros((10,))

answer=np.diag(np.full(10,1))

label = 0
image = []
lala = []
average_cost=0
correct=0
leanring_rate=0.01
new_correct=0
row1=[]

testdata = pd.read_csv(r'D:\Dataset\Testdata.csv')

# Load all parameters
weights1_2 = load_weights_from_csv('weights1_2.csv')
weights2_3 = load_weights_from_csv('weights2_3.csv')
weights3_4 = load_weights_from_csv('weights3_4.csv')

biases1_2 = load_biases_from_csv('biases1_2.csv')
biases2_3 = load_biases_from_csv('biases2_3.csv')
biases3_4 = load_biases_from_csv('biases3_4.csv')



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

for n in range(9998):
    row1 = testdata.iloc[n]
    pixels=row1[1:785].values.astype(np.float32) / 255.0 
    a1 = sigmoid(np.dot(pixels, weights1_2) + biases1_2)
    a2 = sigmoid(np.dot(a1, weights2_3) + biases2_3)
    a3 = softmax(np.dot(a2, weights3_4) + biases3_4)
    row1 = testdata.iloc[n]
    label1 = int(row1[0])
    
    prediction1 = np.argmax(a3)
    if prediction1 == label1:
        new_correct += 1
print("The new accuraccy is", new_correct*100/9998)