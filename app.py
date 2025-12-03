import io
import base64

from flask import Flask, request, render_template
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2

# ------------ CONFIG ------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 224

# IMPORTANT: this must match your training class order
# If in Kaggle: train_ds.classes == ['NORMAL', 'PNEUMONIA'],
# then 0 = Normal, 1 = Pneumonia.
LABEL_MAP = {0: "Normal", 1: "Pneumonia"}

# ------------ MODEL LOADING ------------
def load_model():
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    try:
        state_dict = torch.load("model.pth", map_location="cpu", weights_only=False)
    except TypeError:
        # For older PyTorch that doesn't support weights_only
        state_dict = torch.load("model.pth", map_location="cpu")

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# ------------ PREPROCESSING ------------
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ------------ GRAD-CAM ------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_backward_hook(backward_hook))

    def generate(self, x, class_idx=None):
        self.model.zero_grad()

        out = self.model(x)

        if class_idx is None:
            class_idx = out.argmax(dim=1).item()

        target = out[0, class_idx]
        target.backward()

        grads = self.gradients[0]   # [C,H,W]
        acts  = self.activations[0] # [C,H,W]
        weights = grads.mean(dim=(1, 2))

        cam = torch.zeros(acts.shape[1:], dtype=torch.float32).to(acts.device)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = torch.relu(cam)
        cam_min, cam_max = cam.min(), cam.max()
        cam = cam - cam_min
        if (cam_max - cam_min) != 0:
            cam = cam / (cam_max - cam_min)

        return cam.detach().cpu().numpy()

    def remove(self):
        for h in self.hooks:
            h.remove()

# Target last conv layer of ResNet-50
target_layer = model.layer4[-1].conv3
grad_cam = GradCAM(model, target_layer)

# ------------ HELPER: PIL -> base64 ------------
def pil_to_b64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# ------------ FLASK APP ------------
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    prob_normal = None
    prob_pneumonia = None
    heatmap_img = None
    original_img = None
    error = None

    if request.method == "POST":
        if "file" not in request.files:
            error = "No file uploaded."
            return render_template("index.html", error=error)

        file = request.files["file"]
        if file.filename == "":
            error = "No file selected."
            return render_template("index.html", error=error)

        try:
            img = Image.open(file).convert("RGB")
        except Exception:
            error = "Could not open image. Please upload a valid image file."
            return render_template("index.html", error=error)

        # Save original for display
        original_img = pil_to_b64(img)

        # Preprocess
        input_tensor = preprocess(img).unsqueeze(0).to(DEVICE)

        # Predict
        with torch.no_grad():
            out = model(input_tensor)
            probs = torch.softmax(out, dim=1)[0].cpu().numpy()

        pred_idx = int(probs.argmax())
        prediction = LABEL_MAP.get(pred_idx, f"Class {pred_idx}")

        prob_normal = float(probs[0])
        prob_pneumonia = float(probs[1])

        # Grad-CAM heatmap
        cam = grad_cam.generate(input_tensor, class_idx=pred_idx)

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = cv2.resize(heatmap, img.size)

        img_np = np.array(img)
        overlay = (0.6 * img_np + 0.4 * heatmap).astype(np.uint8)

        overlay_pil = Image.fromarray(overlay)
        heatmap_img = pil_to_b64(overlay_pil)

    return render_template(
        "index.html",
        prediction=prediction,
        prob_normal=prob_normal,
        prob_pneumonia=prob_pneumonia,
        heatmap_img=heatmap_img,
        original_img=original_img,
        error=error,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
