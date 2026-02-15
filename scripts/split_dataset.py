import os
import random
import shutil
from pathlib import Path


# CONFIG

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

SRC_DIR = Path("images")
OUT_DIR = Path("data_split")

CLASSES = ["clean", "medium", "litter"]

random.seed(SEED)


# CREATE OUTPUT FOLDERS

for split in ["train", "val", "test"]:
    for cls in CLASSES:
        (OUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)


# SPLIT FUNCTION

def split_class_images(class_name):
    img_dir = SRC_DIR / class_name
    images = list(img_dir.glob("*"))

    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]

    for img in train_imgs:
        shutil.copy(img, OUT_DIR / "train" / class_name / img.name)

    for img in val_imgs:
        shutil.copy(img, OUT_DIR / "val" / class_name / img.name)

    for img in test_imgs:
        shutil.copy(img, OUT_DIR / "test" / class_name / img.name)

    print(f"{class_name}: "
          f"train={len(train_imgs)}, "
          f"val={len(val_imgs)}, "
          f"test={len(test_imgs)}")

# RUN SPLIT

for cls in CLASSES:
    split_class_images(cls)

print("\n Dataset split completed successfully!")
