# Data description

The release folder contains the sentence-level datasets for the lexical simplification
task.

## Files

- `En-Dataset.csv` – English sentences  
- `Fr-Dataset.csv` – French sentences  

Both files share the same schema:

| Column        | Type    | Description                                                  |
|---------------|---------|--------------------------------------------------------------|
| `ID`          | string  | Article / document identifier                                |
| `Name`        | string  | Article title                                                |
| `Sentence`    | string  | Individual sentence text (in EN or FR)                      |
| `Label`       | int     | 0 = sentence annotated as **simple**, 1 = **complex**   |
| `LengthWords` | int     | Number of tokens in the sentence                            |
| `LengthChars` | int     | Number of characters in the sentence                        |

Notes:

- Articles are typically split into multiple sentences sharing the same `ID`
  and `Name`.
- `LengthWords` and `LengthChars` are provided to help you explore basic
  relationships between length and simplification.

You may create additional processed files (e.g. parallel corpora) under
`data/processed/`.



# iDEM Research Task (EN/FR)

Text complexity classification with anchor-based training and ACC calibration for prevalence estimation under label noise (Vikidia/Wikipedia datasets).

## Overview

This repository contains solutions for the iDEM candidate task on English/French sentence complexity classification.

### Project Structure

```text
idem-candidate-task/
├── src/
│   ├── 01_data_overview.py                    # Task 0: Data overview and cleaning
│   ├── 02_estimate_simplified_proportion.py   # Task 1: Prevalence estimation
│   └── 03_free_analysis.py                    # Task 2: Transformer classifier (EN, Kaggle)
├── data/
│   ├── En-Dataset.csv                         # Raw English data (auto-downloaded)
│   ├── Fr-Dataset.csv                         # Raw French data (auto-downloaded)
│   ├── En-Dataset_cleaned.csv                 # Cleaned English data (generated)
│   └── Fr-Dataset_cleaned.csv                 # Cleaned French data (generated)
├── features/
│   ├── en_full_features.csv                   # English features (from Kaggle)
│   └── fr_full_features.csv                   # French features (from Kaggle)
├── results/                                   # Output summaries and visualizations (Task 1)
├── output/                                    # Duplicate reports & overview plots (Task 0)
├── feature_extraction_kaggle_multi.py         # Feature extraction script (runs on Kaggle)
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/AD2000X/idem-candidate-task.git
cd idem-candidate-task

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Scripts

### 1. `01_data_overview.py` — Data Overview & Cleaning (Task 0)

**Purpose:** Download raw data, perform cleaning, and generate statistics/visualizations.

**Files Used**

| File | Location |
|------|----------|
| En-Dataset.csv | data/ |
| Fr-Dataset.csv | data/ |

**Data Acquisition**

No manual download required. The script automatically downloads missing files from the official GitHub release.

**Usage**

```bash
python src/01_data_overview.py
```

**What it does**

- Checks whether data/En-Dataset.csv and data/Fr-Dataset.csv exist.

- If missing, automatically downloads them from the original iDEM GitHub release:
  - .../releases/download/data/En-Dataset.csv
  - .../releases/download/data/Fr-Dataset.csv

- Performs basic cleaning:
  - Remove NaNs and blank sentences
  - Detect and remove internal duplicates (V-V, W-W)
  - Remove cross-dataset leakage from Vikidia (V-W)

- Computes descriptive statistics and saves overview plots.

**Outputs**

- Cleaned datasets:
  - data/En-Dataset_cleaned.csv
  - data/Fr-Dataset_cleaned.csv

- Duplicate reports:
  - output/en_duplicates.csv
  - output/fr_duplicates.csv

- Overview plots:
  - output/*.png

---

### 2. `02_estimate_simplified_proportion.py` — Prevalence Estimation (Task 1)

**Purpose:** Estimate the true proportion of simple sentences using anchor-based training and ACC calibration.

**Files Used**

| File | Location | Source |
|------|----------|--------|
| en_full_features.csv | features/ | Kaggle dataset |
| fr_full_features.csv | features/ | Kaggle dataset |

Each feature file contains:

- Metadata: ID, Sentence, Label, etc.
- Length-based features (e.g. LengthWords, LengthChars)
- Lexical and syntactic features (e.g. TTR, clause counts, dependency depth)

**Data Acquisition**

Manual download required. These feature files are not downloaded automatically.

**Steps:**

1. Open the Kaggle dataset:
   https://www.kaggle.com/datasets/ad2000x/data-features

2. Click Download to get the .zip file.

3. Unzip locally and copy:
   - en_full_features.csv
   - fr_full_features.csv

   into:

   ```text
   idem-candidate-task/features/
   ```

**Usage**

```bash
python src/02_estimate_simplified_proportion.py
```

**What it does**

- Loads the full feature tables for EN and FR.

- Preprocesses features:
  - Drop metadata and label columns
  - IQR clipping for outliers
  - Remove zero-variance and highly correlated features

- Selects anchor samples:
  - Simple anchors: Vikidia (Label=0) with LengthWords ≤ Q1
  - Complex anchors: Wikipedia (Label=1) with LengthWords ≥ Q3

- Trains Logistic Regression and Random Forest on anchors (with hyperparameter search).

- Uses the best model to predict P(Simple) for all sentences.

- Computes ACC (Adjusted Classify and Count) calibration:
  - Estimates TPR/FPR on anchors via cross-validation
  - Applies p_true = (p_pred - FPR) / (TPR - FPR)

- Produces:
  - Adjusted true simple-sentence prevalence for EN and FR
  - Proportion of Vikidia-like simple sentences inside Wikipedia (soft / hard estimates)

**Outputs**

- Per-language results:
  - results/prevalence_estimation_en.json
  - results/prevalence_estimation_fr.json
  - results/prevalence_estimation_en.png
  - results/prevalence_estimation_fr.png

- Cross-language summary:
  - results/prevalence_estimation_summary.csv

---

### 3. `03_free_analysis.py` — Transformer Classifier (Task 2)

**Purpose:** Train a small Transformer-based classifier (DistilBERT) for sentence complexity, using an anchor-based strategy (English only).
The script is designed to run on a Kaggle GPU notebook.

**Files Used**

| File | Location | Source |
|------|----------|--------|
| En-Dataset_cleaned.csv | Kaggle /input | Output of Task 0 |

**Data Acquisition**

Designed to run on Kaggle. The cleaned English dataset is uploaded as a Kaggle dataset and attached to a notebook.

**Steps:**

1. Run Task 0 locally:

   ```bash
   python src/01_data_overview.py
   ```

   This produces data/En-Dataset_cleaned.csv.

2. Upload En-Dataset_cleaned.csv to Kaggle and create a dataset, e.g.:
   https://www.kaggle.com/datasets/ad2000x/dataset-cleaned

3. In a Kaggle GPU Notebook:
   - Click Add data.
   - Attach the dataset ad2000x/dataset-cleaned.

The script expects:

```python
INPUT_DIR = Path("/kaggle/input/dataset-cleaned")
DATA_FILE = "En-Dataset_cleaned.csv"
# Reads: /kaggle/input/dataset-cleaned/En-Dataset_cleaned.csv
```

**Local Execution (optional)**

To run locally, download the Kaggle dataset and adjust INPUT_DIR to point to your local path.

---

## Pipeline Summary

```text
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: Data Cleaning (01_data_overview.py)                    │
│  ─────────────────────────────────────────────                  │
│  Input:  En-Dataset.csv, Fr-Dataset.csv (auto-downloaded)       │
│  Output: En-Dataset_cleaned.csv, Fr-Dataset_cleaned.csv         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: Feature Extraction (feature_extraction_kaggle_multi.py)│
│  ─────────────────────────────────────────────────────────────  │
│  Input:  En/Fr cleaned datasets                                 │
│  Output: en_full_features.csv, fr_full_features.csv             │
│  Note:   Run on Kaggle, then download features to /features     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: Prevalence Estimation (02_estimate_simplified_...)     │
│  ─────────────────────────────────────────────────────────────  │
│  Input:  en_full_features.csv, fr_full_features.csv             │
│  Output: Prevalence estimates, calibration stats, plots         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Step 4: Transformer Classifier (03_free_analysis.py)           │
│  ─────────────────────────────────────────────────────────────  │
│  Input:  En-Dataset_cleaned.csv (Kaggle dataset)                │
│  Output: Fine-tuned model, predictions, evaluation metrics      │
│  Note:   Requires Kaggle GPU                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Kaggle Datasets

| Dataset | URL | Contents |
|---------|-----|----------|
| Full Features (EN/FR) | https://www.kaggle.com/datasets/ad2000x/data-features | en_full_features.csv, fr_full_features.csv |
| Cleaned English Dataset | https://www.kaggle.com/datasets/ad2000x/dataset-cleaned | En-Dataset_cleaned.csv |

---

## Label Definition

| Label | Meaning | Source |
|-------|---------|--------|
| 0 | Simple | Vikidia (simplified encyclopedia) |
| 1 | Complex | Wikipedia (standard encyclopedia) |

---

## Methods (Task 1)

### Anchor-Based Training

**Simple anchors:**
- Vikidia sentences (Label = 0)
- LengthWords ≤ Q1 (short sentences)

**Complex anchors:**
- Wikipedia sentences (Label = 1)
- LengthWords ≥ Q3 (long sentences)

### ACC Calibration

Adjusted Classify and Count (ACC) is used to correct the biased predicted prevalence:

```text
p_true = (p_predicted - FPR) / (TPR - FPR)
```

where p_predicted is the fraction of sentences predicted as simple (P(Simple) ≥ 0.5),
and TPR/FPR are estimated from cross-validated predictions on anchor data.

### Cleaning Strategy (Task 0)

1. Remove NaN and blank sentences.
2. Detect Vikidia internal duplicates (V–V) and remove them.
3. Detect Wikipedia internal duplicates (W–W) and remove them.
4. Detect cross-dataset duplicates (V–W); remove leaked Wikipedia sentences from Vikidia.

---

## License

This project is prepared for the iDEM candidate task evaluation.

---

## Author

Po-Hsuan Chang