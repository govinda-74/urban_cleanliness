from pathlib import Path
import torch
import numpy as np
from torchvision import datasets, transforms, models

# CONFIG 
DATA_DIR = Path("data_split/test")
MODEL_PATH = Path("runs/baselineA_resnet18/best_model_state.pt")
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TRANSFORMS

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])


# LOAD DATA
ds = datasets.ImageFolder(DATA_DIR, transform=tfms)
loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)

class_names = ds.classes


# LOAD MODEL
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()


# FIND MISCLASSIFICATIONS
errors = []

with torch.no_grad():
    for idx, (x, y_true) in enumerate(loader):
        x = x.to(DEVICE)
        logits = model(x)
        y_pred = torch.argmax(logits, dim=1).item()
        y_true = y_true.item()

        if y_pred != y_true:
            img_path, _ = ds.samples[idx]
            errors.append((img_path, class_names[y_true], class_names[y_pred]))


# PRINT RESULTS

for e in errors:
    print(f"{e[0]}  |  true={e[1]}  pred={e[2]}")

print(f"\nTotal misclassified images: {len(errors)}")
