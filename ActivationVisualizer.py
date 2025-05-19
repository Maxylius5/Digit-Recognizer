import tkinter as tk
from tkinter import Canvas
import numpy as np

class ActivationVisualizer:
    def __init__(self, activations):
        self.activations = activations  # Dictionary with keys: "Layer 1", "Layer 2", ...
        self.window = tk.Toplevel()
        self.window.title("Neural Network Activations")
        self.canvas = Canvas(self.window, width=1000, height=1200, bg="white")
        self.canvas.pack()

        self.draw_activations()

    def draw_activations(self):
        layer_gap = 100
        neuron_radius = 5
        vertical_spacing = 7
        max_neurons = max(len(a) for a in self.activations.values())
        y_center =400

        for i, (layer_name, activations) in enumerate(self.activations.items()):
            x = 100 + i * layer_gap
            self.canvas.create_text(x, 20, text=layer_name, font=("Arial", 12, "bold"))

            total_neurons = len(activations)
            y_offset = y_center - (total_neurons * (neuron_radius + vertical_spacing)) // 2

            for j, value in enumerate(activations):
                gray = int((1 - value) * 255)  # 0 = black, 1 = white
                fill = f"#{gray:02x}{gray:02x}{gray:02x}"
                y = y_offset + j * (neuron_radius + vertical_spacing)
                self.canvas.create_oval(
                    x - neuron_radius, y - neuron_radius,
                    x + neuron_radius, y + neuron_radius,
                    fill=fill, outline="black"
                )
