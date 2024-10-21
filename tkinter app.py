import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageOps
import tkinter as tk
from tkinter import *

# Load the pre-trained CNN model
model = load_model(r"C:\Users\HP\OneDrive\Desktop\ML PROJECTS\DIGIT RECOGNISER BY CNN\digit_model.keras")

# Function to predict digit
def predict_digit(img):
    # Resize the image to 28x28 pixels as required by the model
    img = img.resize((28, 28))
    img = img.convert('L')  # Convert to grayscale
    img = ImageOps.invert(img)  # Invert colors
    img = np.array(img)

    # Normalize image
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    # Predict digit
    result = model.predict([img])[0]
    return np.argmax(result), max(result)

# Tkinter GUI for digit drawing and recognition
class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Digit Recognizer")
        self.geometry("600x400")
        self.configure(bg="#f0f0f0")

        # Create frames for better layout
        self.canvas_frame = Frame(self, bg="#f0f0f0")
        self.canvas_frame.grid(row=0, column=0, padx=10, pady=10)

        self.controls_frame = Frame(self, bg="#f0f0f0")
        self.controls_frame.grid(row=1, column=0, padx=10, pady=10)

        self.result_frame = Frame(self, bg="#f0f0f0", relief="sunken", bd=2)
        self.result_frame.grid(row=0, column=1, padx=10, pady=10, rowspan=2, sticky="n")

        # Set up canvas for drawing
        self.canvas = tk.Canvas(self.canvas_frame, width=300, height=300, bg='white', relief="sunken", bd=2)
        self.canvas.pack()

        # Create a blank PIL image to store the drawn content
        self.image = PILImage.new("RGB", (300, 300), (255, 255, 255))  # Using PILImage alias
        self.draw = ImageDraw.Draw(self.image)

        # Bind mouse motion to the painting action
        self.canvas.bind("<B1-Motion>", self.paint)

        # Buttons for clearing the canvas and predicting
        self.button_clear = tk.Button(self.controls_frame, text="Clear", command=self.clear_canvas, width=10, bg="#d9534f", fg="white", font=('Helvetica', 12, 'bold'))
        self.button_clear.grid(row=1, column=0, pady=5, padx=5)

        self.button_predict = tk.Button(self.controls_frame, text="Predict", command=self.get_prediction, width=10, bg="#5cb85c", fg="white", font=('Helvetica', 12, 'bold'))
        self.button_predict.grid(row=1, column=1, pady=5, padx=5)

        # Label to show the prediction result
        self.result_label = tk.Label(self.result_frame, text="Draw a digit and click Predict", font=('Helvetica', 18), fg='black', bg='#f0f0f0', wraplength=250, justify="center")
        self.result_label.pack(padx=10, pady=20, expand=True)

    # Function to draw on canvas
    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=10)
        self.draw.line([x1, y1, x2, y2], fill="black", width=10)

    # Function to clear the canvas
    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 300, 300], fill="white")
        self.result_label.config(text="Draw a digit and click Predict", fg='blue')

    # Function to get prediction from the CNN model
    def get_prediction(self):
        # Get a copy of the in-memory image and predict
        img = self.image.copy()
        digit, acc = predict_digit(img)
        self.result_label.config(text=f"Predicted: {digit}\nAccuracy: {acc*100:.2f}%", fg='green')

# Run the Tkinter app
if __name__ == "__main__":
    app = App()
    app.mainloop()
