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

def main():
# CONFIG

    SEED = 42
    DATA_DIR = Path("data_split")   # change if needed
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_WORKERS = 2

    EPOCHS_FROZEN = 6
    EPOCHS_FINETUNE = 12

    LR_FROZEN = 1e-3
    LR_FINETUNE = 1e-4
    WEIGHT_DECAY = 1e-4

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    OUT_DIR = Path("runs/baselineA_resnet18")
    OUT_DIR.mkdir(parents=True, exist_ok=True)



    # REPRODUCIBILITY
    
    def set_seed(seed: int):
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
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=7),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])


    
    # DATA LOADERS
    
    def make_loader(split: str, tfms, shuffle: bool):
        ds = datasets.ImageFolder(DATA_DIR / split, transform=tfms)
        loader = DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=shuffle,
            num_workers=NUM_WORKERS, pin_memory=True
        )
        return ds, loader

    train_ds, train_loader = make_loader("train", train_tfms, shuffle=True)
    val_ds, val_loader     = make_loader("val", eval_tfms, shuffle=False)
    test_ds, test_loader   = make_loader("test", eval_tfms, shuffle=False)

    class_names = train_ds.classes
    num_classes = len(class_names)

    assert num_classes == 3, f"Expected 3 classes, found: {class_names}"
    print("Classes:", class_names)
    print("Counts:",
        "train =", len(train_ds),
        "val =", len(val_ds),
        "test =", len(test_ds))

    # Save metadata
    meta = {
        "seed": SEED,
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs_frozen": EPOCHS_FROZEN,
        "epochs_finetune": EPOCHS_FINETUNE,
        "lr_frozen": LR_FROZEN,
        "lr_finetune": LR_FINETUNE,
        "weight_decay": WEIGHT_DECAY,
        "device": DEVICE,
        "classes": class_names,
    }
    (OUT_DIR / "config.json").write_text(json.dumps(meta, indent=2))


    # MODEL (ResNet-18 pretrained)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()


    
    # TRAIN / EVAL HELPERS
    
    def train_one_epoch(model, loader, optimizer):
        model.train()
        running_loss = 0.0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        return running_loss / len(loader.dataset)

    @torch.no_grad()
    def evaluate(model, loader):
        model.eval()
        all_preds, all_true = [], []
        running_loss = 0.0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            running_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_true.append(y.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_true  = np.concatenate(all_true)

        avg_loss = running_loss / len(loader.dataset)
        acc = float((all_preds == all_true).mean())
        macro_f1 = float(f1_score(all_true, all_preds, average="macro"))
        return avg_loss, acc, macro_f1, all_true, all_preds


    def save_confusion_matrix(cm, labels, outpath):
        fig = plt.figure()
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix (Test)")
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.yticks(range(len(labels)), labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")
        plt.tight_layout()
        fig.savefig(outpath, dpi=200)
        plt.close(fig)



    # PHASE 1: FROZEN BACKBONE
    
    print("\nPHASE 1: Train classifier head only (frozen backbone)")
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR_FROZEN, weight_decay=WEIGHT_DECAY)

    best_val_f1 = -1.0
    best_state = None
    history = []

    for epoch in range(1, EPOCHS_FROZEN + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, optimizer)
        va_loss, va_acc, va_f1, _, _ = evaluate(model, val_loader)
        dt = time.time() - t0

        row = {"phase": "frozen", "epoch": epoch, "train_loss": tr_loss,
            "val_loss": va_loss, "val_acc": va_acc, "val_macro_f1": va_f1, "sec": dt}
        history.append(row)
        print(f"[{epoch}/{EPOCHS_FROZEN}] train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
            f"val_acc={va_acc:.3f} val_f1={va_f1:.3f} time={dt:.1f}s")

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    
    # PHASE 2: FINE-TUNE last block + head
    
    print("\nPHASE 2: Fine-tune layer4 + classifier head")
    model.load_state_dict(best_state)

    for p in model.parameters():
        p.requires_grad = False
    for p in model.layer4.parameters():
        p.requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True

    params = list(model.layer4.parameters()) + list(model.fc.parameters())
    optimizer = torch.optim.Adam(params, lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    for epoch in range(1, EPOCHS_FINETUNE + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, optimizer)
        va_loss, va_acc, va_f1, _, _ = evaluate(model, val_loader)
        scheduler.step(va_f1)
        dt = time.time() - t0

        row = {"phase": "finetune", "epoch": epoch, "train_loss": tr_loss,
            "val_loss": va_loss, "val_acc": va_acc, "val_macro_f1": va_f1, "sec": dt}
        history.append(row)
        print(f"[{epoch}/{EPOCHS_FINETUNE}] train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
            f"val_acc={va_acc:.3f} val_f1={va_f1:.3f} time={dt:.1f}s")

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Save best model
    torch.save(best_state, OUT_DIR / "best_model_state.pt")
    (OUT_DIR / "history.json").write_text(json.dumps(history, indent=2))

    print(f"\n Best val macro-F1 = {best_val_f1:.3f}")
    print(f"Saved model to: {OUT_DIR / 'best_model_state.pt'}")


    
    # FINAL TEST EVALUATION 
    
    print("\nFINAL: Test evaluation (do this once and report it)")
    model.load_state_dict(best_state)
    te_loss, te_acc, te_f1, y_true, y_pred = evaluate(model, test_loader)

    print(f"TEST loss={te_loss:.4f} acc={te_acc:.3f} macro_f1={te_f1:.3f}")

    cm = confusion_matrix(y_true, y_pred)
    save_confusion_matrix(cm, class_names, OUT_DIR / "confusion_matrix_test.png")

    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    (OUT_DIR / "classification_report_test.txt").write_text(report)

    print("\nClassification report saved to:", OUT_DIR / "classification_report_test.txt")
    print("Confusion matrix image saved to:", OUT_DIR / "confusion_matrix_test.png")
    pass

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
