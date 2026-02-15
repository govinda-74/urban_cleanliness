
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

## Dataset Splitting Script(split_dataset.py)
 - This script creates a reproducible train/validation/test split for the Urban Cleanliness image dataset while preserving class balance.
  
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
