#  Handwritten Digit Recognizer – Neural Network (NumPy & PyTorch)

This project implements a neural network that recognizes handwritten digits from the **MNIST** dataset. It includes two versions:

1.  A **NumPy-only implementation**, building the neural network completely from scratch (no ML libraries)
   
    This implementation has 2 hidden layers between the input and output layer with 20 neurons each. It achieved a 95.5% accuracy on the testing data after just 20 epochs
2.  A **PyTorch version**, using a more efficient framework for training and inference

It also features a simple GUI (created with hekp from chatGPT, as that was not the focus of this project) that allows the user to draw digits and get real-time predictions.

---

##  Features

- Two neural network implementations:
  - Pure NumPy (manual forward/backward pass and training loop)
  - PyTorch (efficient training and model management)
- MNIST dataset preprocessed and loaded manually
- Trained weights can be saved and reloaded
- Interactive **28x28 GUI** input window (draw a digit with your mouse)
- Optional visualizations for internal activations of hidden layers

---




The format from the CVS file is

    label, pix-11, pix-12, pix-13, ...

Refer to [MNIST in CSV](https://pjreddie.com/projects/mnist-in-csv/)

