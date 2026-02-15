import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image


# CONFIG

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = Path("runs/baselineA_resnet18/best_model_state.pt")

INPUT_DIR = Path("gradcam_samples")
OUTPUT_DIR = Path("figures/gradcam")

IMG_SIZE = 224


CLASSES = ["clean", "litter", "medium"]


# CREATE OUTPUT DIRECTORIES

for sub in ["correct", "wrong"]:
    (OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)

# IMAGE TRANSFORMS
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

# LOAD MODEL
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# GRAD-CAM hooks and storage

activations = []
gradients = []

def forward_hook(module, input, output):
    activations.append(output)

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

target_layer = model.layer4
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# GRAD-CAM FUNCTION

def generate_gradcam(img_tensor, class_idx):
    activations.clear()
    gradients.clear()

    output = model(img_tensor)
    score = output[:, class_idx]

    model.zero_grad()
    score.backward()

    grads = gradients[0]        # [1, C, H, W]
    acts  = activations[0]      # [1, C, H, W]

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1)

    cam = F.relu(cam)
    cam = cam.squeeze().detach().cpu().numpy()

    cam -= cam.min()
    cam /= cam.max() + 1e-8

    return cam


# PROCESS IMAGES

def process_folder(subfolder):
    for img_name in os.listdir(INPUT_DIR / subfolder):
        img_path = INPUT_DIR / subfolder / img_name

        # Load image
        img_pil = Image.open(img_path).convert("RGB")
        img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

        # Forward pass
        with torch.no_grad():
            logits = model(img_tensor)
        pred_class = logits.argmax(dim=1).item()

        # Grad-CAM
        cam = generate_gradcam(img_tensor, pred_class)
        cam = cv2.resize(cam, img_pil.size)

        # Heatmap overlay
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        img_np = np.array(img_pil)
        overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

        # Save
        out_name = img_name.replace(".png", "_cam.png")
        out_path = OUTPUT_DIR / subfolder / out_name
        cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        print(f"Saved Grad-CAM: {out_path}")


# MAIN

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

    process_folder("correct")
    process_folder("wrong")

    print("\n Grad-CAM generation completed successfully.")
