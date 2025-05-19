import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import os
from PIL import Image, ImageDraw, ImageOps
import tkinter as tk

'''
def load_model(path):
    model = MNISTNet().to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")
    return model

model_path = r'D:\Dataset\model_parameters\mnist_model.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
canvas_size = 500
brush_size = 10


class DigitRecognizer:
    def __init__(self, model):
        self.model = model
        self.window = tk.Tk()
        self.window.title("MNIST Digit Recognizer")

        self.canvas = tk.Canvas(self.window, width=canvas_size, height=canvas_size, bg='white')
        self.canvas.pack()

        self.image = Image.new("L", (canvas_size, canvas_size), 255)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-3>", self.clear_canvas)

        self.predict_btn = tk.Button(self.window, text="Predict", command=self.predict_digit)
        self.predict_btn.pack()

        self.result_label = tk.Label(self.window, text="", font=("Arial", 20))
        self.result_label.pack()

        self.window.mainloop()

    def paint(self, event):
        x1, y1 = event.x - brush_size, event.y - brush_size
        x2, y2 = event.x + brush_size, event.y + brush_size
        self.canvas.create_oval(x1, y1, x2, y2, fill='black')
        self.draw.ellipse([x1, y1, x2, y2], fill=0)

    def clear_canvas(self, event=None):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, canvas_size, canvas_size], fill=255)
        self.result_label.config(text="")

    def predict_digit(self):
        # Convert to MNIST format (28x28, inverted)
        img_resized = self.image.resize((28, 28), Image.BILINEAR)
        img_inverted = ImageOps.invert(img_resized)
        
        # Convert to numpy array and normalize
        pixels = np.array(img_inverted).astype(np.float32).reshape(784) / 255.0
        
        # Get prediction
        digit, probabilities = predict(self.model, pixels)
        top_prob = max(probabilities)
        
        # Display results
        self.result_label.config(text=f"Prediction: {digit}")
        self.confidence_label.config(text=f"Confidence: {top_prob*100:.1f}%")
        
        # Print all probabilities (for debugging)
        print("\nPrediction probabilities:")
        for i, prob in enumerate(probabilities):
            print(f"{i}: {prob*100:.1f}%")


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
    

def load_model(path):
        model = MNISTNet().to(device)
        model.load_state_dict(torch.load(path))
        model.eval()
        print(f"Model loaded from {path}")
        return model
def predict(model, pixel_data):
    with torch.no_grad():
        tensor = torch.tensor(pixel_data, device=device).unsqueeze(0)  # Add batch dimension
        output = model(tensor)
        _, predicted = torch.max(output.data, 1)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        return predicted.item(), probabilities.squeeze().tolist()



if __name__ == "__main__":
    
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please train the model first or check the path.")
    else:
        model = load_model(model_path)
        app = DigitRecognizer(model)    '''



from ActivationVisualizer import ActivationVisualizer 

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = r'D:\Dataset\model_parameters\mnist_model.pth'
canvas_size = 500
brush_size = 15

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
    

def load_model(path):
    model = MNISTNet().to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")
    return model

def predict(model, pixel_data):
    with torch.no_grad():
        tensor = torch.tensor(pixel_data, device=device).unsqueeze(0)  # Add batch dimension
        output = model(tensor)
        _, predicted = torch.max(output.data, 1)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        return predicted.item(), probabilities.squeeze().tolist()

class DigitRecognizer:
    def __init__(self, model):
        self.model = model
        self.window = tk.Tk()
        self.window.title("MNIST Digit Recognizer")
        
        # Create canvas
        self.canvas = tk.Canvas(self.window, width=canvas_size, height=canvas_size, bg='white')
        self.canvas.pack()
        
        # Create PIL image for drawing
        self.image = Image.new("L", (canvas_size, canvas_size), 255)
        self.draw = ImageDraw.Draw(self.image)
        
        # Bind mouse events
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_prev_point)
        self.prev_point = None
        
        # Create buttons
        button_frame = tk.Frame(self.window)
        button_frame.pack(fill=tk.X)
        
        self.clear_btn = tk.Button(button_frame, text="Clear", command=self.clear_canvas)
        self.clear_btn.pack(side=tk.LEFT, padx=10)
        
        self.predict_btn = tk.Button(button_frame, text="Predict", command=self.predict_digit)
        self.predict_btn.pack(side=tk.LEFT, padx=10)
        
        # Result display
        self.result_label = tk.Label(self.window, text="Draw a digit (0-9)", font=("Arial", 24))
        self.result_label.pack()
        
        # Confidence display
        self.confidence_label = tk.Label(self.window, text="", font=("Arial", 14))
        self.confidence_label.pack()
        
        self.window.mainloop()
    
    def paint(self, event):
        x, y = event.x, event.y
        
        # Draw on canvas
        if self.prev_point:
            self.canvas.create_line(self.prev_point[0], self.prev_point[1], x, y, 
                                  width=brush_size, fill='black', capstyle=tk.ROUND, smooth=True)
            self.draw.line([self.prev_point[0], self.prev_point[1], x, y], 
                          fill=0, width=brush_size)
        
        self.prev_point = (x, y)
    
    def reset_prev_point(self, event):
        self.prev_point = None
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, canvas_size, canvas_size], fill=255)
        self.result_label.config(text="Draw a digit (0-9)")
        self.confidence_label.config(text="")
    
    def predict_digit(self):
        # Convert to MNIST format (28x28, inverted)
        img_resized = self.image.resize((28, 28), Image.BILINEAR)
        img_inverted = ImageOps.invert(img_resized)
        
        # Convert to numpy array and normalize
        pixels = np.array(img_inverted).astype(np.float32).reshape(784) / 255.0
        
        # Get prediction
        digit, probabilities = predict(self.model, pixels)
        top_prob = max(probabilities)
        
        # Display results
        self.result_label.config(text=f"Prediction: {digit}")
        self.confidence_label.config(text=f"Confidence: {top_prob*100:.1f}%")
        
        img_resized = self.image.resize((28, 28), Image.BILINEAR)
        img_inverted = ImageOps.invert(img_resized)
        pixels = np.array(img_inverted).astype(np.float32).reshape(784) / 255.0
        tensor = torch.tensor(pixels, device=device).unsqueeze(0)

    # Collect intermediate activations manually
        with torch.no_grad():
            a1 = self.model.sigmoid(self.model.fc1(tensor))
            a2 = self.model.sigmoid(self.model.fc2(a1))
            a3 = self.model.sigmoid(self.model.fc3(a2))
            out = self.model.fc4(a3)
            prediction = torch.argmax(out).item()
            probabilities = torch.nn.functional.softmax(out, dim=1).squeeze().tolist()

    # Show in label
        self.result_label.config(text=f"Prediction: {prediction}")
        self.confidence_label.config(text=f"Confidence: {max(probabilities)*100:.1f}%")

    # Create activation window
        act_dict = {
            
            "Layer 2": a2.squeeze().cpu().numpy(),
            "Layer 3": a3.squeeze().cpu().numpy(),
            "Output": np.array(probabilities),
        }
        ActivationVisualizer(act_dict)

        # Print all probabilities (for debugging)
        print("\nPrediction probabilities:")
        for i, prob in enumerate(probabilities):
            print(f"{i}: {prob*100:.1f}%")

if __name__ == "__main__":
    # Load trained model
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please train the model first or check the path.")
    else:
        model = load_model(model_path)
        app = DigitRecognizer(model)