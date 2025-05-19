import pandas as pd

import numpy as np 

traindata = pd.read_csv(r'D:\Dataset\traindata.csv')


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def deriv_sigmoid(x):
    s=sigmoid(x)
    return s*(1-s)

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

for epoch in range(100):

    for n in range(59995):

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


        delta3 = (result - answer[label] ) * activation3
        delta2 = (weights3_4.T @ delta3) * deriv_sigmoid(activation2)
        delta1 = (weights2_3.T @ delta2) * deriv_sigmoid(image)

print("Training succesfully complited")




testdata = pd.read_csv(r'D:\Dataset\Testdata.csv')

for n in range(9995):

    image, activation2 = compute_activations1_2(testdata.iloc[n], weights1_2, biases1_2)
    activation3 = compute_activations2_3(activation2, weights2_3, biases2_3)
    result = compute_activations2_3(activation3, weights3_4, biases3_4)
    row = testdata.iloc[n]
    label = int(row[0])
    result=softmax(result)
    prediction = np.argmax(result)

    if prediction == label:
        correct += 1
print("The accuraccy is", correct*100/9995)

# cost function with respect to the weight(of the previos neuron) - is :    
#   C/a34=(activations2_3-result) * 2
#   a/z= sigmoid(z) # z= w * a-1 - b
#   z/w= a-1
#   cost function for weight (C/w) = activations1_2(z/w) * sigmoid(z)(a/z) * (activations2_3-result)*2  -- C/a

# * with respect to the bias:
#   z/b=1
#   a/z= sigmoid(z) # z= w * a-1 - b
#   z/w= a-1
#   Cf bias = sigmoid(z)(a/z) * (activations2_3-result)*2 -- C/b

# * with respect to the activation of the previous neuron:
#   z/a=w
#   a/z= sigmoid(z) # z= w * a-1 - b
#   z/w= a-1
#   Cf bias = sigmoid(z)(a/z) * (activations2_3-result)*2 -- C/b


'''
def learning_backPropagation(w2,a1,a2,b2, goal):
    cost_function_w = a1 * sigmoid(weighted_sum(a1,w2,b2)) * 2 * (a2 - goal)
    cost_function_b = sigmoid(weighted_sum(a1,w2,b2)) * 2 * (a2 - goal)
    cost_function_a = w2 * sigmoid(weighted_sum(a1,w2,b2)) * 2 * (a2 - goal)
    return cost_function_a, cost_function_b, cost_function_w

def trying_again(a2,a1,w1,b1, true_label):
    #loss or cost = mean squared error 
    #derivative_cost=2*(a1 - true_label)
    dc_dw = np.outer(a1, a1 * (1 - a1) , (a2 - true_label))
    dc_db = np.outer(a1 * (1 - a1), a2 - true_label)'''




