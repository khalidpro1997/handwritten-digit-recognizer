import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Define the model (must match train_model.py)
class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.layers(x)

# Load the trained model
model = DigitClassifier()
model.load_state_dict(torch.load("mnist_model.pth", map_location=torch.device("cpu")))
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognizer")

        self.canvas = tk.Canvas(root, width=200, height=200, bg="white")
        self.canvas.pack()

        self.label = tk.Label(root, text="Draw a digit and click Predict", font=("Helvetica", 16))
        self.label.pack()

        self.predict_button = tk.Button(root, text="Predict", command=self.predict)
        self.predict_button.pack()

        self.clear_button = tk.Button(root, text="Clear", command=self.clear)
        self.clear_button.pack()

        self.image = Image.new("L", (200, 200), color=255)
        self.draw = ImageDraw.Draw(self.image)

        self.last_x, self.last_y = None, None
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_pos)

    def paint(self, event):
        if self.last_x and self.last_y:
            # Draw smooth line from last point to current
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                    width=20, fill="black", capstyle=tk.ROUND, smooth=True)
            self.draw.line([self.last_x, self.last_y, event.x, event.y], fill=0, width=20)
        self.last_x = event.x
        self.last_y = event.y

    def reset_pos(self, event):
        self.last_x, self.last_y = None, None

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 200, 200], fill=255)
        self.label.config(text="Draw a digit and click Predict")

    def predict(self):
        # Prepare image
        img = self.image.copy()
        img = ImageOps.invert(img)
        img = img.resize((28, 28))
        img_tensor = transform(img).unsqueeze(0)

        # Predict
        with torch.no_grad():
            output = model(img_tensor)
            pred = output.argmax(dim=1).item()
            self.label.config(text=f"Predicted Digit: {pred}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
