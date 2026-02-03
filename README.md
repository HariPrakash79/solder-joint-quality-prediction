# Solder Joint Quality Prediction (XRay Baseline)

Baseline ML pipeline to classify solder joint quality from XRay images using the HellaStudy-of-LEDs dataset. The goal is to turn measured void-rate data into defect labels and train an image classifier that can scale inspection in manufacturing.

## Why this matters
Solder joint defects (voids/cracks) reduce reliability in semiconductor and PCB products and can lead to costly field failures. Automated inspection helps improve yield, reduce rework, and standardize quality control.

## What is included
- `inspect_dataset.ipynb`: data inspection, CSV parsing, and label creation.
- `train_model.ipynb`: baseline training (224x224 CNN, panel-level split).
- `extract_pdf.py`: helper to extract text from the reference paper.

## Dataset
The dataset is **not** included in this repo due to size and licensing. Download the "HellaStudy-of-LEDs" dataset from Kaggle and place folders in the project root:
- `XRay/`
- `SAM/`
- `TTA/`
- `TTA-Raw/`
- `CrackVoid Ratios/`

The key label file used in the baseline is:
- `CrackVoid Ratios/Xray Void Ratio.csv`

## Quickstart (conda)
1) Create environment:
```
conda create -n soldercracks python=3.10 -y
conda install -n soldercracks -y ipykernel pandas numpy scikit-learn tensorflow
```
2) Register kernel:
```
python -m ipykernel install --user --name soldercracks --display-name "Python (soldercracks)"
```
3) Run notebooks in order:
- `inspect_dataset.ipynb`
- `train_model.ipynb`

## Baseline approach
1) Read `Xray Void Ratio.csv` and convert `Void rate` to numeric.
2) Create a binary label using a percentile threshold (default: top 25% = defect).
3) Parse image filenames to map them to CSV rows.
4) Panel-level train/val/test split (reduces leakage).
5) Train a small CNN on 224x224 images.

Example run: ~0.75 test accuracy (will vary by split and threshold).

## Next steps
- Replace baseline CNN with MobileNet (better accuracy, efficient for edge).
- Add data augmentation and class weighting.
- Explore regression (predict void rate directly).
- Add explainability (Grad-CAM).
- Multi-modal learning: combine XRay + SAM + TTA.

## Notes
Large datasets, environments, and reference PDFs are excluded via `.gitignore`.
