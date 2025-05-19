import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import pandas as pd
import os

# ========== LOAD SAVED MODEL ==========
save_dir = r'D:\Dataset\model_parameters'

def load_weights(filename):
    return pd.read_csv(os.path.join(save_dir, filename), header=None).values

def load_biases(filename):
    return pd.read_csv(os.path.join(save_dir, filename), header=None).values.flatten()

weights1_2 = load_weights('weights1_2.csv')
weights2_3 = load_weights('weights2_3.csv')
weights3_4 = load_weights('weights3_4.csv')
biases1_2 = load_biases('biases1_2.csv')
biases2_3 = load_biases('biases2_3.csv')
biases3_4 = load_biases('biases3_4.csv')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

def predict(pixels):
    a1 = sigmoid(np.dot(pixels, weights1_2) + biases1_2)
    a2 = sigmoid(np.dot(a1, weights2_3) + biases2_3)
    a3 = softmax(np.dot(a2, weights3_4) + biases3_4)
    return np.argmax(a3)

# ========== GUI PART ==========
canvas_size = 500  # 280x280, will be scaled to 28x28
brush_size = 10

class DigitRecognizer:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Draw a digit (0-9)")

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
        img_resized = self.image.resize((28, 28))
        img_inverted = ImageOps.invert(img_resized)
        pixels = np.array(img_inverted).astype(np.float32).reshape(784) / 255.0
        prediction = predict(pixels)
        self.result_label.config(text=f"Prediction: {prediction}", )

DigitRecognizer()
