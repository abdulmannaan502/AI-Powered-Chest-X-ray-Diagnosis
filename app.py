import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import gradio as gr

# ------------------------
# CONFIG
# ------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["Normal", "Pneumonia"]

# ------------------------
# MODEL
# ------------------------
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)

state_dict = torch.load("model.pth", map_location=DEVICE)
model.load_state_dict(state_dict)

model.to(DEVICE)
model.eval()

# ------------------------
# TRANSFORMS
# ------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# ------------------------
# GRAD CAM
# ------------------------
class GradCAM:
    def __init__(self, model, layer):
        self.model=model
        self.layer=layer
        self.grad=None
        self.act=None

        layer.register_forward_hook(self.fw_hook)
        layer.register_backward_hook(self.bw_hook)

    def fw_hook(self, m,i,o):
        self.act=o.detach()

    def bw_hook(self, m,gi,go):
        self.grad=go[0].detach()

    def generate(self, img_tensor, class_idx):
        self.model.zero_grad()
        out = self.model(img_tensor)
        out[0, class_idx].backward()

        w = self.grad.mean((2,3))[0]
        cam = (w[:,None,None] * self.act[0]).sum(0)
        cam = torch.relu(cam)

        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        return cam

target_layer = model.layer4[-1]
cam_gen = GradCAM(model, target_layer)

# ------------------------
# INFERENCE FUNCTION
# ------------------------
def diagnose(image):
    image = image.convert("RGB")
    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out[0], dim=0)
        pred = probs.argmax().item()

    cam = cam_gen.generate(x, pred)
    cam_img = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    cam_img = cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB)
    cam_img = cv2.resize(cam_img, image.size)

    overlay = (0.6*np.array(image) + 0.4*cam_img).astype("uint8")

    return CLASS_NAMES[pred], float(probs[0]), float(probs[1]), Image.fromarray(overlay)

# ------------------------
# GRADIO UI
# ------------------------
iface = gr.Interface(
    fn=diagnose,
    inputs=gr.Image(type="pil", label="Upload Chest X-ray"),
    outputs=[
        gr.Text(label="Prediction"),
        gr.Number(label="Confidence Normal"),
        gr.Number(label="Confidence Pneumonia"),
        gr.Image(label="Grad-CAM Heatmap")
    ],
)

iface.launch()
