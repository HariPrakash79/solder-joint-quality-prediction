# Solder Joint Quality Prediction (XRay Baseline + MobileNet)

Baseline ML pipeline to classify solder joint quality from XRay images using the HellaStudy-of-LEDs dataset. The goal is to turn measured void-rate data into defect labels and train image classifiers that can scale inspection in manufacturing.

## Why this matters
Solder joint defects (voids/cracks) reduce reliability in semiconductor and PCB products and can lead to costly field failures. Automated inspection helps improve yield, reduce rework, and standardize quality control.

## What is included
- `inspect_dataset.ipynb`: data inspection, CSV parsing, and label creation.
- `train_baseline.ipynb`: baseline CNN training (224x224, panel-level split).
- `train_mobilenet.ipynb`: MobileNetV2 transfer learning + fine-tuning.
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
conda install -n soldercracks -y ipykernel pandas numpy scikit-learn tensorflow matplotlib
```
2) Register kernel:
```
python -m ipykernel install --user --name soldercracks --display-name "Python (soldercracks)"
```
3) Run notebooks in order:
- `inspect_dataset.ipynb`
- `train_baseline.ipynb`
- `train_mobilenet.ipynb`

## Baseline approach
1) Read `Xray Void Ratio.csv` and convert `Void rate` to numeric.
2) Create a binary label using a threshold (current: `THRESHOLD = 0.04`).
3) Parse image filenames to map them to CSV rows.
4) Panel-level train/val/test split (reduces leakage).
5) Train a small CNN on 224x224 images.

## Evaluation
- **Label threshold** defines what counts as defect in the CSV (current: `0.04`).
- **Decision threshold** controls how strict predictions are (default `0.5`, tuned via PR curve).
- We track confusion matrix + PR curve to balance defect recall vs false alarms.

## Current results (latest run)
Baseline CNN (filters `[8, 16, 32]`, dropout `0.3`):
- Test accuracy: ~0.77
- Mild overfitting but stable validation trends

MobileNetV2 (fine-tuned):
- Test accuracy: ~0.77 at default threshold 0.5
- With decision threshold ~0.127, defect recall improves to ~0.95 (more false alarms)

## Results snapshot (latest run)
| Model | Label threshold | Decision threshold | Test accuracy | Defect recall | Notes |
|---|---|---|---|---|---|
| Baseline CNN | 0.04 | 0.50 | ~0.77 | ~0.68 | Mild overfitting, stable pipeline |
| MobileNetV2 (fine tuned + class weights) | 0.04 | 0.50 | ~0.78 | ~0.53 | Default cutoff misses defects |
| MobileNetV2 (fine tuned + class weights) | 0.04 | 0.165 | ~0.80 | ~0.95 | Threshold tuned for defect recall |

Notes:
- **Label threshold** defines defect in the CSV (`void_rate >= 0.04`).
- **Decision threshold** is the probability cutoff for predictions.


## Next steps
- Tune decision threshold for target recall/precision tradeoff.
- Try class weights to increase defect recall without moving threshold too low.
- Explore regression (predict void rate directly) as a comparison.
- Add explainability (Grad-CAM).
- Multi-modal learning: combine XRay + SAM + TTA.

- Build a regression model with EfficientNet (predict void rate directly).
- Compare regression thresholded accuracy vs classification models.
## Notes
Large datasets, environments, and reference PDFs are excluded via `.gitignore`.
