# Urban Cleanliness Classification (Clean / Medium / Litter)
## Labeling Criteria

Clean:
- No visible litter on the ground
- Pavement/road appears tidy

Medium:
- Small scattered litter
- Trash bins may be present
- Litter does not dominate the scene

Litter:
- Multiple visible waste items
- Garbage piles or widespread trash
- Litter visually dominates the ground

Labels reflect human perception and are subjective.

Classify urban street images into **Clean**, **Medium**, and **Litter** using a CNN (ResNet-18).  
This project compares **transfer learning (pretrained on ImageNet)** vs **training from scratch**, and uses **Grad-CAM** for explainability.


## Project Overview
- **Task:** 3-class image classification (Clean / Medium / Litter)
- **Models:**
  - ResNet-18 (Pretrained, fine-tuned in 2 phases)
  - ResNet-18 (From scratch)
- **Explainability:** Grad-CAM heatmaps for correct and misclassified samples

## Dataset
- **Source:** - https://www.kaggle.com/datasets/nw8vqlafd/street-classification-dataset
- **dataset:** - download it from the source and reorganized manually into 3 classes as it contain some extra data also.Seperated the medium liiter images from more litter ones and created seperate class for it. 
   
 
https://drive.google.com/drive/folders/12ojbFjkev9edPna561bLI7ngTk-tWnJ4?usp=drive_link
- **Total images:** 750 (balanced)
- **Per class:** 250 images each
- **Split:** 70% train / 15% val / 15% test



---

## Repository Structure
```text
urban_cleanliness/
├─ srcipt/                # training + utility scripts 
├─ gradcam_samples/       # saved Grad-CAM examples (correct/ + wrong/)
├─ results/               # confusion matrices + classification reports
├─ .gitignore
└─ README.md


# Dataset Splitting Script(split_dataset.py)
After running the script, the following structure is created:
  
data_split/
├── train/
│   ├── clean/
│   ├── medium/
│   └── litter/
├── val/
│   ├── clean/
│   ├── medium/
│   └── litter/
└── test/
    ├── clean/
    ├── medium/
    └── litter/

##Pretrained Model: ResNet-18 (Transfer Learning)

We use **ResNet-18 pretrained on ImageNet** as the primary baseline model. 
Instead of training from random initialization, we leverage learned visual features from a large-scale dataset (ImageNet) and adapt them to the urban cleanliness task.

### Why Pretrained?
ResNet-18 pretrained on ImageNet already learns:
- Low-level features (edges, textures)
- Mid-level patterns (shapes, object parts)
- High-level semantic representations

This helps the model:
- Converge faster
- Perform better on small datasets
- Generalize better to unseen data

---

### Training Strategy (Two-Phase Fine-Tuning)

We fine-tune the model in **two phases**:

#### Phase 1 — Frozen Backbone
- Freeze all convolutional layers
- Train only the final fully connected (classification) layer
- Learning rate: 1e-3
- Purpose: Adapt classifier to 3 classes (Clean / Medium / Litter)

#### Phase 2 — Fine-Tuning
- Unfreeze the last residual block (layer4)
- Train layer4 + classifier
- Lower learning rate: 1e-4
- Purpose: Adapt high-level features to urban cleanliness domain

Model selection is based on **validation macro-F1 score**.

---

### Loss & Optimization
- Loss function: Cross-Entropy Loss
- Optimizer: Adam
- Regularization: Weight decay (1e-4)
- Learning rate scheduling: ReduceLROnPlateau (based on validation F1)

---

### Performance (Test Set)

| Metric | Value |
|--------|-------|
| Accuracy | 0.886 |
| Macro F1-score | 0.887 |

The pretrained model significantly outperforms the model trained from scratch, demonstrating the benefit of transfer learning for small datasets.
