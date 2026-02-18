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


## Dataset Splitting Script(split_dataset.py)
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
