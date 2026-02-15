from pathlib import Path
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

print("FILE IS BEING EXECUTED")


# CONFIG

def main():
    print("MAIN FUNCTION STARTED")

    SEED = 42
    DATA_DIR = Path("data_split")
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_WORKERS = 2

    EPOCHS = 40
    LR = 1e-3
    WEIGHT_DECAY = 1e-4

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    OUT_DIR = Path("runs/from_scratch_resnet18")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    
    # REPRODUCIBILITY
    
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    set_seed(SEED)


    # TRANSFORMS
    
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(7),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    
    # DATA LOADERS
    
    def make_loader(split, tfms, shuffle):
        ds = datasets.ImageFolder(DATA_DIR / split, transform=tfms)
        loader = DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )
        return ds, loader

    train_ds, train_loader = make_loader("train", train_tfms, True)
    val_ds, val_loader = make_loader("val", eval_tfms, False)
    test_ds, test_loader = make_loader("test", eval_tfms, False)

    class_names = train_ds.classes
    num_classes = len(class_names)

    print("Classes:", class_names)
    print("Train:", len(train_ds), "Val:", len(val_ds), "Test:", len(test_ds))

    
    # MODEL (FROM SCRATCH)
    
    model = models.resnet18(weights=None)  # NO PRETRAINING
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    
    # TRAIN / EVAL FUNCTIONS

    def train_one_epoch():
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        return total_loss / len(train_loader.dataset)

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        all_preds, all_true = [], []
        total_loss = 0

        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(y.cpu().numpy())

        avg_loss = total_loss / len(loader.dataset)
        acc = np.mean(np.array(all_preds) == np.array(all_true))
        macro_f1 = f1_score(all_true, all_preds, average="macro")

        return avg_loss, acc, macro_f1, all_true, all_preds


    # TRAINING LOOP
    
    best_val_f1 = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        start = time.time()

        train_loss = train_one_epoch()
        val_loss, val_acc, val_f1, _, _ = evaluate(val_loader)

        scheduler.step(val_f1)

        elapsed = time.time() - start

        print(f"[{epoch}/{EPOCHS}] "
            f"Train Loss={train_loss:.4f} | "
            f"Val Loss={val_loss:.4f} | "
            f"Val Acc={val_acc:.3f} | "
            f"Val F1={val_f1:.3f} | "
            f"{elapsed:.1f}s")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), OUT_DIR / "best_model_state.pt")

    # Save training history
    (OUT_DIR / "history.json").write_text(json.dumps(history, indent=2))

    print("\nBest Validation F1:", best_val_f1)

    
    # FINAL TEST EVALUATION
    
    model.load_state_dict(torch.load(OUT_DIR / "best_model_state.pt"))

    test_loss, test_acc, test_f1, y_true, y_pred = evaluate(test_loader)

    print("\nTEST RESULTS")
    print("Accuracy:", round(test_acc, 3))
    print("Macro F1:", round(test_f1, 3))

    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix - From Scratch")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(class_names)), class_names)
    plt.yticks(range(len(class_names)), class_names)
    plt.colorbar()

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix_test.png")
    plt.close()

    report = classification_report(y_true, y_pred, target_names=class_names)
    (OUT_DIR / "classification_report_test.txt").write_text(report)

    print("\nSaved results in:", OUT_DIR)
    pass

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
