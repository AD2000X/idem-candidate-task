"""
iDEM Task 1 — Estimate the true proportion of simple sentences
============================================================================

This script estimates the true proportion of simple sentences in the dataset,
accounting for label noise using anchor-based training and ACC calibration.

Label Definition (Used in this script):
    - Label = 0: Simple (Vikidia-style)
    - Label = 1: Complex (Wikipedia-style)

Naive estimate implementation:
    - Naive proportion = #(Label = 0) / total

Pipeline:
    1. Naive Estimate: Direct label proportion (baseline)
    2. Anchor Selection: Use extreme samples for clean training data
    3. Model Training: LR + RF with full ML pipeline
    4. Full Prediction: Predict P(Simple) for all sentences
    5. ACC Calibration: Adjust for classifier bias
    6. Final Estimates: True proportion + Wikipedia internal simple ratio
    7. Additional Analysis: Noise candidates, anchor quality, and stratified prevalence

Usage:
    python 02_estimate_simplified_proportion.py
"""

import gc
import json
import random
import shutil
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings('ignore')


# Configuration
BASE_DIR = Path(__file__).parent
FEATURES_DIR = BASE_DIR / "features"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"

FEATURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Required local files (no download - must exist locally)
# Now using only feature files which contain all necessary data
FEATURE_FILES = {
    "en": "en_full_features.csv",
    "fr": "fr_full_features.csv"
}

# Columns to drop from features (metadata + target + anchor-selection-only)
DROP_COLUMNS = [
    'Index', 'ID', 'Name', 'Sentence',
    'Label',
    'LengthWords', 'LengthChars',
    'source'
]

# Correlation threshold for feature removal
CORRELATION_THRESHOLD = 0.95

# Random seed for reproducibility
RANDOM_SEED = 42

# ACC calibration settings
ACC_CV_FOLDS = 5

# High confidence threshold for Wikipedia analysis (hard estimate)
HIGH_CONF_THRESHOLD = 0.8

# High confidence threshold for mislabel candidate mining
MISLABEL_HIGH_CONF_THRESHOLD = 0.9


# Reproducibility
def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


# File Detection
def find_file_recursive(
    filename: str,
    search_dir: Path,
    max_depth: int = 3
) -> Optional[Path]:
    """Recursively search for a file in directory tree."""
    target = search_dir / filename
    if target.exists():
        return target

    if max_depth > 0:
        try:
            for subdir in search_dir.iterdir():
                if subdir.is_dir() and not subdir.name.startswith('.'):
                    result = find_file_recursive(filename, subdir, max_depth - 1)
                    if result:
                        return result
        except PermissionError:
            pass

    return None


def ensure_local_files_exist() -> None:
    """Check that all required local files exist."""
    print("\n" + "=" * 60)
    print("CHECKING LOCAL FILES")
    print("=" * 60)

    required_files = [
        (FEATURES_DIR, FEATURE_FILES["en"]),
        (FEATURES_DIR, FEATURE_FILES["fr"]),
    ]

    for target_dir, filename in required_files:
        target_path = target_dir / filename

        if target_path.exists():
            print(f"  [OK] {target_path}")
            continue

        # Search recursively
        print(f"  [SEARCH] Looking for {filename}...")
        found = find_file_recursive(filename, BASE_DIR)

        if found:
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(found, target_path)
            print(f"  [COPIED] {found} -> {target_path}")
            continue

        # Not found - raise error
        raise FileNotFoundError(
            f"\n  [ERROR] Missing required file: {filename}\n"
            f"  Please place it in: {target_dir}\n"
        )

    print("\nAll required files found.")


# Data Loading
def load_features(lang: str) -> pd.DataFrame:
    """
    Load pre-extracted features.
    This file contains all necessary data including metadata, labels, and features.
    """
    filename = FEATURE_FILES[lang]
    filepath = FEATURES_DIR / filename

    print(f"\nLoading features: {filepath}")

    with open(filepath, encoding='utf-8') as f:
        total_rows = sum(1 for _ in f) - 1

    chunks = []
    with tqdm(total=total_rows, desc=f"Reading {filename}",
              ncols=80, unit=" rows") as pbar:
        for chunk in pd.read_csv(filepath, chunksize=50000):
            chunks.append(chunk)
            pbar.update(len(chunk))

    features_df = pd.concat(chunks, ignore_index=True)

    # Add source column based on ID prefix
    features_df['source'] = features_df['ID'].apply(
        lambda x: 'wiki' if str(x).startswith('wiki-') else 'viki'
    )

    return features_df


# Part 1: Naive Estimate
def compute_naive_proportion(df: pd.DataFrame) -> float:
    """
    Compute naive estimate: Count(Label=0) / Total.

    Label Definition:
        - Label = 0: Simple (Vikidia-style)
        - Label = 1: Complex (Wikipedia-style)
    """
    total = len(df)
    simple_count = (df['Label'] == 0).sum()

    if total == 0:
        raise ValueError("No rows in dataset")

    return simple_count / total


# Part 2: Feature Preprocessing
def prepare_features(
    features_df: pd.DataFrame
) -> Tuple[pd.DataFrame, List[str]]:
    print("\nPreparing features...")

    # Keep LengthWords temporarily for anchor selection
    cols_to_drop_now = [c for c in DROP_COLUMNS if c != 'LengthWords']
    cols_to_drop_now = [c for c in cols_to_drop_now if c in features_df.columns]

    features_clean = features_df.drop(columns=cols_to_drop_now, errors='ignore')

    print(f"  Dropped columns: {cols_to_drop_now}")
    print(f"  Remaining columns: {list(features_clean.columns)}")

    return features_clean, features_clean.columns.tolist()


def handle_outliers(
    features_df: pd.DataFrame,
    multiplier: float = 3.0
) -> pd.DataFrame:
    print("\nHandling outliers (IQR clipping)...")

    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    n_clipped = 0

    for col in numeric_cols:
        Q1 = features_df[col].quantile(0.25)
        Q3 = features_df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR

        outliers = ((features_df[col] < lower) | (features_df[col] > upper)).sum()
        if outliers > 0:
            features_df[col] = features_df[col].clip(lower, upper)
            n_clipped += 1

    print(f"  Clipped outliers in {n_clipped} columns")

    return features_df


def preprocess_features(features_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    print("\nPreprocessing features...")

    original_count = len(features_df.columns)

    # Remove blank columns
    features_df = features_df.loc[:, features_df.notna().any()]

    # Remove zero-variance columns (excluding LengthWords to keep anchor feature)
    variance = features_df.drop(columns=['LengthWords'], errors='ignore').var()
    zero_var = variance[variance == 0].index.tolist()
    if zero_var:
        print(f"  Removed zero-variance: {zero_var}")
        features_df = features_df.drop(columns=zero_var)

    # Fill NaN with median
    nan_cols = features_df.columns[features_df.isna().any()].tolist()
    if nan_cols:
        print(f"  Filling NaN in: {nan_cols}")
        for col in nan_cols:
            features_df[col] = features_df[col].fillna(features_df[col].median())

    print(f"  Features: {original_count} -> {len(features_df.columns)}")

    return features_df, features_df.columns.tolist()


def remove_high_correlation(
    features_df: pd.DataFrame,
    threshold: float = CORRELATION_THRESHOLD,
    protected_cols: List[str] = None
) -> pd.DataFrame:
    print(f"\nRemoving high correlation (threshold={threshold})...")

    # Protect certain columns from removal (e.g., LengthWords needed for anchor selection)
    if protected_cols is None:
        protected_cols = ['LengthWords']

    corr_matrix = features_df.corr().abs()
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    cols_to_drop = set()
    for col in upper_tri.columns:
        correlated = upper_tri.index[upper_tri[col] > threshold].tolist()
        cols_to_drop.update(correlated)

    # Remove protected columns from drop list
    cols_to_drop = cols_to_drop - set(protected_cols)

    if cols_to_drop:
        print(f"  Removing {len(cols_to_drop)} columns: {cols_to_drop}")
        features_df = features_df.drop(columns=list(cols_to_drop))
    else:
        print("  No highly correlated features found")

    return features_df


# ============================================================
# Part 2: Anchor Selection
# ============================================================

def select_anchors(
    df: pd.DataFrame,
    features_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, pd.Index, Dict]:
    """
    Select clean anchor samples for training.

    Label Definition:
        - Label = 0: Simple (Vikidia-style)
        - Label = 1: Complex (Wikipedia-style)

    Simple Anchor: Vikidia (Label=0) with LengthWords <= Q1
    Complex Anchor: Wikipedia (Label=1) with LengthWords >= Q3
    """
    print("\n" + "=" * 60)
    print("ANCHOR SELECTION")
    print("=" * 60)

    # Get LengthWords from features
    if 'LengthWords' not in features_df.columns:
        raise ValueError("LengthWords column required for anchor selection")

    length_words = features_df['LengthWords']

    # Calculate quartiles
    q1 = length_words.quantile(0.25)
    q3 = length_words.quantile(0.75)

    print(f"\nLengthWords quartiles: Q1={q1:.1f}, Q3={q3:.1f}")

    # Simple anchors: short Vikidia sentences (Label=0)
    simple_mask = (df['Label'] == 0) & (length_words <= q1)
    n_simple = simple_mask.sum()

    # Complex anchors: long Wikipedia sentences (Label=1)
    complex_mask = (df['Label'] == 1) & (length_words >= q3)
    n_complex = complex_mask.sum()

    print(f"\nAnchor selection:")
    print(f"  Simple anchors (Vikidia/Label=0, LengthWords <= {q1:.1f}): {n_simple:,}")
    print(f"  Complex anchors (Wikipedia/Label=1, LengthWords >= {q3:.1f}): {n_complex:,}")

    # Combine anchors
    anchor_mask = simple_mask | complex_mask
    anchor_indices = df.index[anchor_mask]

    training_features = features_df.drop(columns=['LengthWords'], errors='ignore')
    X_anchor = training_features.loc[anchor_mask].values
    y_anchor = df.loc[anchor_mask, 'Label'].values

    print(f"  Total anchor samples: {len(anchor_indices):,}")
    print(f"  Class distribution: Simple(0)={sum(y_anchor==0):,}, Complex(1)={sum(y_anchor==1):,}")

    anchor_info = {
        'q1': float(q1),
        'q3': float(q3),
        'n_simple_anchors': int(n_simple),
        'n_complex_anchors': int(n_complex),
        'total_anchors': int(len(anchor_indices))
    }

    return X_anchor, y_anchor, anchor_indices, anchor_info


# Part 2: Model Building and Training
def build_lr_pipeline() -> Pipeline:
    """Build Logistic Regression pipeline."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=RANDOM_SEED,
            n_jobs=-1
        ))
    ])


def build_rf_pipeline() -> Pipeline:
    """Build Random Forest pipeline."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            class_weight='balanced',
            random_state=RANDOM_SEED,
            n_jobs=-1
        ))
    ])


def tune_model(
    pipeline: Pipeline,
    param_distributions: Dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_name: str
) -> Tuple[Pipeline, Dict]:
    """Tune model hyperparameters using RandomizedSearchCV."""
    print(f"\n  Tuning {model_name}...")

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)

    search = RandomizedSearchCV(
        pipeline, param_distributions,
        n_iter=12,
        cv=cv, scoring='f1',
        random_state=RANDOM_SEED,
        n_jobs=-1, verbose=0
    )

    search.fit(X_train, y_train)

    # Validation metrics
    val_pred = search.predict(X_val)
    val_proba = search.predict_proba(X_val)[:, 1]

    results = {
        'best_params': search.best_params_,
        'cv_f1': float(search.best_score_),
        'val_f1': float(f1_score(y_val, val_pred, pos_label=0)),
        'val_roc_auc': float(roc_auc_score(y_val, val_proba))
    }

    print(f"    Best params: {search.best_params_}")
    print(f"    CV F1: {results['cv_f1']:.4f} | Val F1: {results['val_f1']:.4f}")

    return search.best_estimator_, results


def train_models(
    X_anchor: np.ndarray,
    y_anchor: np.ndarray
) -> Tuple[Pipeline, str, Dict]:
    """Train LR and RF models on anchor data, return best model."""
    print("\n" + "=" * 60)
    print("MODEL TRAINING (on anchor samples)")
    print("=" * 60)

    # Split anchors into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_anchor, y_anchor,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y_anchor
    )

    print(f"\nAnchor split: Train={len(X_train):,}, Val={len(X_val):,}")

    results = {}

    # Logistic Regression
    lr_params = {
        'classifier__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'classifier__penalty': ['l2'],
        'classifier__solver': ['lbfgs', 'saga']
    }
    lr_pipeline, lr_results = tune_model(
        build_lr_pipeline(), lr_params,
        X_train, y_train, X_val, y_val,
        "Logistic Regression"
    )
    results['logistic_regression'] = lr_results

    # Random Forest
    rf_params = {
        'classifier__n_estimators': [50, 100, 150, 200],
        'classifier__max_depth': [5, 10, 15, 20, 25, None],
        'classifier__min_samples_split': [2, 5, 10, 20],
        'classifier__min_samples_leaf': [1, 2, 4]
    }
    rf_pipeline, rf_results = tune_model(
        build_rf_pipeline(), rf_params,
        X_train, y_train, X_val, y_val,
        "Random Forest"
    )
    results['random_forest'] = rf_results

    # Select best model by validation F1 (Simple=0 as positive)
    if rf_results['val_f1'] >= lr_results['val_f1']:
        best_model = rf_pipeline
        best_name = 'random_forest'
    else:
        best_model = lr_pipeline
        best_name = 'logistic_regression'

    print(f"\nBest model: {best_name} (Val F1: {results[best_name]['val_f1']:.4f})")

    return best_model, best_name, results


# Anchor quality / leakage sanity check
def label_shuffle_sanity_check(
    X_anchor: np.ndarray,
    y_anchor: np.ndarray,
    max_samples: int = 50000
) -> Dict:
    """
    Sanity check for potential leakage:
    Shuffle labels and measure how well a RandomForest can fit.
    If F1 remains very high, this suggests serious leakage.
    """
    print("\nRunning label-shuffle sanity check on anchors...")

    n = len(y_anchor)
    if n == 0:
        return {'n_samples_used': 0, 'mean_f1': 0.0, 'std_f1': 0.0, 'n_splits': 0}

    if n > max_samples:
        rng = np.random.RandomState(RANDOM_SEED)
        idx = rng.choice(n, size=max_samples, replace=False)
        X_sub = X_anchor[idx]
        y_sub = y_anchor[idx]
    else:
        X_sub = X_anchor
        y_sub = y_anchor

    y_shuffled = np.random.permutation(y_sub)

    model = build_rf_pipeline()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED + 1)
    scores = cross_val_score(
        model, X_sub, y_shuffled,
        cv=cv, scoring='f1', n_jobs=-1
    )

    mean_f1 = float(scores.mean())
    std_f1 = float(scores.std())

    print(f"  Used samples: {len(y_sub):,} / {n:,}")
    print(f"  Label-shuffle RF F1 (Simple=0 as positive): mean={mean_f1:.4f}, std={std_f1:.4f}")

    return {
        'n_samples_used': int(len(y_sub)),
        'n_total_anchor_samples': int(n),
        'mean_f1': mean_f1,
        'std_f1': std_f1,
        'n_splits': 3
    }


def analyze_anchor_quality(
    df: pd.DataFrame,
    anchor_idx: pd.Index,
    y_anchor: np.ndarray
) -> Dict:
    """
    Summarise anchor quality: ratio, length distribution, and source breakdown.
    """
    print("\n" + "=" * 60)
    print("ANCHOR QUALITY SUMMARY")
    print("=" * 60)

    total_n = len(df)
    anchor_n = len(anchor_idx)
    anchor_ratio = anchor_n / total_n if total_n > 0 else 0.0

    length_global = df['LengthWords']
    length_anchor = df.loc[anchor_idx, 'LengthWords']

    source_anchor = df.loc[anchor_idx, 'source'].value_counts().to_dict()

    print(f"  Anchor ratio: {anchor_n:,} / {total_n:,} = {anchor_ratio:.4f}")
    print(f"  LengthWords mean (global): {length_global.mean():.2f}, std: {length_global.std():.2f}")
    print(f"  LengthWords mean (anchors): {length_anchor.mean():.2f}, std: {length_anchor.std():.2f}")
    print(f"  Anchor source breakdown: {source_anchor}")

    # Basic class balance in anchors
    simple_anchors = int((y_anchor == 0).sum())
    complex_anchors = int((y_anchor == 1).sum())
    print(f"  Anchor label breakdown: Simple(0)={simple_anchors:,}, Complex(1)={complex_anchors:,}")

    anchor_quality = {
        'anchor_ratio': float(anchor_ratio),
        'n_total': int(total_n),
        'n_anchor': int(anchor_n),
        'lengthwords_global_mean': float(length_global.mean()),
        'lengthwords_global_std': float(length_global.std()),
        'lengthwords_anchor_mean': float(length_anchor.mean()),
        'lengthwords_anchor_std': float(length_anchor.std()),
        'anchor_source_breakdown': source_anchor,
        'anchor_label_breakdown': {
            'simple_0': simple_anchors,
            'complex_1': complex_anchors
        }
    }

    return anchor_quality


# Part 3: Full Prediction
def predict_full_dataset(
    model: Pipeline,
    features_df: pd.DataFrame
) -> np.ndarray:
    """Predict P(Simple) for all sentences."""
    print("\n" + "=" * 60)
    print("FULL DATASET PREDICTION")
    print("=" * 60)

    # Remove LengthWords for prediction
    X_full = features_df.drop(columns=['LengthWords'], errors='ignore').values

    print(f"\nPredicting probabilities for {len(X_full):,} sentences...")

    # Note: predict_proba returns [P(class_0), P(class_1)]
    # Label=0 is Simple, so P(Simple) = predict_proba[:, 0]
    probabilities = model.predict_proba(X_full)[:, 0]

    print(f"  Mean P(Simple): {probabilities.mean():.4f}")
    print(f"  Std P(Simple): {probabilities.std():.4f}")

    return probabilities


# Part 2: ACC Calibration
def compute_acc_calibration(
    model: Pipeline,
    X_anchor: np.ndarray,
    y_anchor: np.ndarray,
    n_folds: int = ACC_CV_FOLDS
) -> Tuple[float, float, Dict]:
    """
    Compute TPR and FPR via cross-validation on anchor data.

    These rates are used for ACC (Adjusted Classify and Count) calibration.

    For Simple class (Label=0):
        - TP: Correctly predicted as Simple (Label=0)
        - FP: Incorrectly predicted as Simple (actually Complex/Label=1)
        - TN: Correctly predicted as Complex (Label=1)
        - FN: Incorrectly predicted as Complex (actually Simple/Label=0)
    """
    print("\n" + "=" * 60)
    print("ACC CALIBRATION (Cross-Validation)")
    print("=" * 60)

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    # Get cross-validated predictions
    y_pred_cv = cross_val_predict(model, X_anchor, y_anchor, cv=cv)

    # Compute confusion matrix with labels=[1, 0] so that Simple(0) is positive class
    # This gives: [[TN, FP], [FN, TP]] where positive = Simple(0)
    cm = confusion_matrix(y_anchor, y_pred_cv, labels=[1, 0])
    tn, fp, fn, tp = cm.ravel()

    # Calculate rates for Simple class (Label=0)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate (Recall for Simple)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate

    print(f"\nConfusion Matrix (Simple=0 as positive, {n_folds}-fold CV):")
    print(f"  TN={tn:,} (Complex->Complex), FP={fp:,} (Complex->Simple)")
    print(f"  FN={fn:,} (Simple->Complex), TP={tp:,} (Simple->Simple)")
    print(f"\nRates for Simple class:")
    print(f"  TPR (True Positive Rate): {tpr:.4f}")
    print(f"  FPR (False Positive Rate): {fpr:.4f}")

    calibration_info = {
        'n_folds': n_folds,
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'tpr': float(tpr),
        'fpr': float(fpr)
    }

    return tpr, fpr, calibration_info


def apply_acc_correction(
    p_pred: float,
    tpr: float,
    fpr: float
) -> float:
    """
    Apply ACC formula to correct predicted proportion.

    Formula: p_true = (p_pred - FPR) / (TPR - FPR)
    """
    denominator = tpr - fpr

    if abs(denominator) < 1e-10:
        print("  [WARNING] TPR ~= FPR, ACC correction undefined. Using p_pred.")
        return p_pred

    p_true = (p_pred - fpr) / denominator

    # Clip to valid range [0, 1]
    p_true = max(0.0, min(1.0, p_true))

    return p_true


# Part 4: Final Estimates
def compute_final_estimates(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    tpr: float,
    fpr: float,
    naive_proportion: float
) -> Dict:
    """
    Compute final prevalence estimates.

    1. Adjusted true proportion (ACC-corrected)
    2. Wikipedia internal Simple-style proportion
    """
    print("\n" + "=" * 60)
    print("FINAL ESTIMATES")
    print("=" * 60)

    # 1. Predicted proportion (before ACC)
    p_pred = (probabilities >= 0.5).mean()

    # 2. ACC-corrected true proportion
    p_true = apply_acc_correction(p_pred, tpr, fpr)

    print(f"\n1. GLOBAL SIMPLE SENTENCE PROPORTION:")
    print(f"   Naive estimate (Label=0 / Total):     {naive_proportion:.4f} ({naive_proportion*100:.2f}%)")
    print(f"   Predicted proportion (P >= 0.5):      {p_pred:.4f} ({p_pred*100:.2f}%)")
    print(f"   ACC-corrected true proportion:        {p_true:.4f} ({p_true*100:.2f}%)")

    # 3. Wikipedia internal analysis
    wiki_mask = df['source'] == 'wiki'
    wiki_probs = probabilities[wiki_mask]
    n_wiki = wiki_mask.sum()

    # Soft count: average probability
    wiki_simple_soft = wiki_probs.mean()

    # Hard count: high confidence threshold
    wiki_simple_hard = (wiki_probs >= HIGH_CONF_THRESHOLD).mean()

    print(f"\n2. WIKIPEDIA INTERNAL SIMPLE-STYLE PROPORTION:")
    print(f"   Wikipedia sentences:                  {n_wiki:,}")
    print(f"   Soft estimate (mean P(Simple)):       {wiki_simple_soft:.4f} ({wiki_simple_soft*100:.2f}%)")
    print(f"   Hard estimate (P >= {HIGH_CONF_THRESHOLD}):            {wiki_simple_hard:.4f} ({wiki_simple_hard*100:.2f}%)")

    estimates = {
        'naive_proportion': float(naive_proportion),
        'predicted_proportion': float(p_pred),
        # Adjusted true proportion of simple sentences
        'acc_corrected_proportion': float(p_true),
        # Wikipedia internal: proportion of Vikidia-like simple sentences inside Wikipedia
        'wikipedia_analysis': {
            'n_sentences': int(n_wiki),
            'soft_estimate': float(wiki_simple_soft),
            'hard_estimate': float(wiki_simple_hard),
            'hard_threshold': float(HIGH_CONF_THRESHOLD)
        },
        # Descriptions for JSON consumers
        'descriptions': {
            'acc_corrected_proportion': (
                'Adjusted true proportion of simple sentences in the full dataset '
                'using ACC calibration.'
            ),
            'wikipedia_soft_estimate': (
                'Proportion of Vikidia-like simple sentences among complex-labelled '
                'Wikipedia sentences (soft estimate: mean P(Simple) over Wikipedia sentences).'
            ),
            'wikipedia_hard_estimate': (
                f'Proportion of Vikidia-like simple sentences among complex-labelled '
                f'Wikipedia sentences (hard estimate: fraction with P(Simple) >= {HIGH_CONF_THRESHOLD}).'
            )
        }
    }

    return estimates


def analyze_full_predictions(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    lang: str,
    threshold: float = 0.5,
    high_conf_mislabel: float = MISLABEL_HIGH_CONF_THRESHOLD
) -> Dict:
    """
    Additional analysis on full dataset:
    - Confusion matrix (overall and by source)
    - Vikidia vs Wikipedia naive + predicted simple proportions
    - Length-bin prevalence profile
    - High-confidence disagreement candidates (potential label noise)
    """
    print("\n" + "=" * 60)
    print("FULL DATASET PREDICTION ANALYSIS")
    print("=" * 60)

    y_true = df['Label'].values
    # Predict label: Simple(0) if P(Simple) >= threshold, else Complex(1)
    y_pred = np.where(probabilities >= threshold, 0, 1)

    # Overall confusion matrix (Simple=0 as positive)
    cm_full = confusion_matrix(y_true, y_pred, labels=[1, 0])
    tn, fp, fn, tp = cm_full.ravel()

    acc = accuracy_score(y_true, y_pred)
    f1_simple = f1_score(y_true, y_pred, pos_label=0)

    print("\nOverall confusion matrix (Simple=0 as positive):")
    print(f"  TN={tn:,} (Complex->Complex), FP={fp:,} (Complex->Simple)")
    print(f"  FN={fn:,} (Simple->Complex), TP={tp:,} (Simple->Simple)")
    print(f"  Accuracy: {acc:.4f}, F1(Simple=0): {f1_simple:.4f}")

    confusion_by_source: Dict[str, Dict[str, int]] = {}
    prevalence_by_source: Dict[str, Dict[str, float]] = {}

    for src in ['viki', 'wiki']:
        mask = (df['source'] == src).values
        n_src = int(mask.sum())
        if n_src == 0:
            continue

        y_true_src = y_true[mask]
        y_pred_src = y_pred[mask]
        probs_src = probabilities[mask]

        cm_src = confusion_matrix(y_true_src, y_pred_src, labels=[1, 0])
        tn_s, fp_s, fn_s, tp_s = cm_src.ravel()

        confusion_by_source[src] = {
            'tn': int(tn_s),
            'fp': int(fp_s),
            'fn': int(fn_s),
            'tp': int(tp_s),
            'n_sentences': n_src
        }

        naive_simple = float((y_true_src == 0).mean())
        soft_pred = float(probs_src.mean())
        hard_pred = float((probs_src >= threshold).mean())

        prevalence_by_source[src] = {
            'n_sentences': n_src,
            'naive_simple_proportion_label0': naive_simple,
            'predicted_simple_soft_mean_prob': soft_pred,
            'predicted_simple_hard_prop_p>=threshold': hard_pred
        }

        print(f"\nSource='{src}' ({n_src:,} sentences):")
        print(f"  Naive simple proportion (Label=0): {naive_simple:.4f}")
        print(f"  Predicted simple (soft, mean P(Simple)): {soft_pred:.4f}")
        print(f"  Predicted simple (hard, P(Simple) >= {threshold}): {hard_pred:.4f}")

    # Length-bin prevalence profile
    length = df['LengthWords']
    bins = [0, 10, 20, 30, 40, np.inf]
    bin_labels = ['<=10', '11-20', '21-30', '31-40', '>=41']
    length_bins = pd.cut(length, bins=bins, labels=bin_labels, include_lowest=True, right=True)

    tmp = pd.DataFrame({
        'length_bin': length_bins,
        'p_simple': probabilities,
        'hard_simple': (probabilities >= threshold).astype(int)
    })

    length_profile_df = tmp.groupby('length_bin', observed=True).agg(
        n_sentences=('p_simple', 'size'),
        mean_p_simple=('p_simple', 'mean'),
        hard_simple_prop=('hard_simple', 'mean')
    ).reset_index()

    length_profile = length_profile_df.to_dict(orient='records')

    print("\nLength-bin prevalence profile (using LengthWords):")
    for row in length_profile:
        print(
            f"  Bin={row['length_bin']}: n={row['n_sentences']}, "
            f"mean P(Simple)={row['mean_p_simple']:.4f}, "
            f"hard simple prop (P>= {threshold})={row['hard_simple_prop']:.4f}"
        )

    # High-confidence disagreement (potential label noise)
    hi = high_conf_mislabel
    lo = 1.0 - hi

    mask_complex_high_simple = (df['Label'] == 1) & (probabilities >= hi)
    mask_simple_low_simple = (df['Label'] == 0) & (probabilities <= lo)
    mask_candidates = mask_complex_high_simple | mask_simple_low_simple

    candidates = df.loc[mask_candidates, ['ID', 'Name', 'Sentence', 'Label', 'source', 'LengthWords']].copy()
    candidates['p_simple'] = probabilities[mask_candidates]
    candidates['pred_label_0.5'] = np.where(candidates['p_simple'] >= threshold, 0, 1)

    candidate_type = np.where(
        mask_complex_high_simple[mask_candidates],
        'complex_label_high_simple_prob',
        'simple_label_low_simple_prob'
    )
    candidates['candidate_type'] = candidate_type

    mislabel_file = RESULTS_DIR / f"mislabel_candidates_{lang}.csv"
    candidates.to_csv(mislabel_file, index=False)

    n_complex_high_simple = int(mask_complex_high_simple.sum())
    n_simple_low_simple = int(mask_simple_low_simple.sum())
    total_candidates = int(mask_candidates.sum())

    print("\nHigh-confidence disagreement candidates (potential label noise):")
    print(f"  complex-label, P(Simple) >= {hi}: {n_complex_high_simple:,}")
    print(f"  simple-label,  P(Simple) <= {lo}: {n_simple_low_simple:,}")
    print(f"  total candidates saved to: {mislabel_file}")

    mislabel_stats = {
        'high_conf_threshold': hi,
        'low_conf_threshold': lo,
        'complex_label_high_simple_prob_count': n_complex_high_simple,
        'simple_label_low_simple_prob_count': n_simple_low_simple,
        'total_candidates': total_candidates,
        'output_file': str(mislabel_file)
    }

    analysis = {
        'threshold_for_hard_simple': threshold,
        'confusion_matrix_full': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
            'n_sentences': int(len(y_true))
        },
        'metrics_full': {
            'accuracy': float(acc),
            'f1_simple_label0': float(f1_simple)
        },
        'confusion_matrix_by_source': confusion_by_source,
        'by_source_prevalence': prevalence_by_source,
        'length_bin_profile': length_profile,
        'mislabel_stats': mislabel_stats
    }

    return analysis


# Visualization
def plot_results(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    estimates: Dict,
    lang: str
) -> None:
    """Generate visualization plots."""
    print("\nGenerating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Prevalence Estimation Results - {lang.upper()}",
        fontsize=14, fontweight='bold'
    )

    # 1. Probability distribution by source
    ax1 = axes[0, 0]
    wiki_probs = probabilities[df['source'] == 'wiki']
    viki_probs = probabilities[df['source'] == 'viki']

    ax1.hist(wiki_probs, bins=50, alpha=0.6, label='Wikipedia (Complex)', color='blue', density=True)
    ax1.hist(viki_probs, bins=50, alpha=0.6, label='Vikidia (Simple)', color='green', density=True)
    ax1.axvline(x=0.5, color='red', linestyle='--', label='Threshold (0.5)')
    ax1.set_xlabel('P(Simple)')
    ax1.set_ylabel('Density')
    ax1.set_title('Probability Distribution by Source')
    ax1.legend()

    # 2. Proportion comparison
    ax2 = axes[0, 1]
    proportions = [
        estimates['naive_proportion'],
        estimates['predicted_proportion'],
        estimates['acc_corrected_proportion']
    ]
    labels = ['Naive\n(Label=0/Total)', 'Predicted\n(P>=0.5)', 'ACC-Corrected\n(True Est.)']
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4']

    bars = ax2.bar(labels, proportions, color=colors)
    ax2.set_ylabel('Proportion')
    ax2.set_title('Simple Sentence Proportion Estimates')
    ax2.set_ylim(0, max(proportions) * 1.2)

    for bar, prop in zip(bars, proportions):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{prop:.2%}', ha='center', va='bottom', fontsize=10)

    # 3. Wikipedia internal analysis
    ax3 = axes[1, 0]
    wiki_est = estimates['wikipedia_analysis']
    wiki_labels = ['Soft Estimate\n(Mean P)', f'Hard Estimate\n(P>={HIGH_CONF_THRESHOLD})']
    wiki_values = [wiki_est['soft_estimate'], wiki_est['hard_estimate']]

    bars = ax3.bar(wiki_labels, wiki_values, color=['#9467bd', '#d62728'])
    ax3.set_ylabel('Proportion')
    ax3.set_title('Wikipedia Internal Simple-Style Proportion')
    ax3.set_ylim(0, max(wiki_values) * 1.4 if max(wiki_values) > 0 else 0.1)

    for bar, val in zip(bars, wiki_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{val:.2%}', ha='center', va='bottom', fontsize=10)

    # 4. Probability boxplot by label
    ax4 = axes[1, 1]
    df_plot = pd.DataFrame({
        'P(Simple)': probabilities,
        'Original Label': df['Label'].map({0: 'Simple/Vikidia (0)', 1: 'Complex/Wikipedia (1)'})
    })
    sns.boxplot(data=df_plot, x='Original Label', y='P(Simple)', ax=ax4)
    ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
    ax4.set_title('Predicted Probability by Original Label')

    plt.tight_layout()

    # Save plot
    plot_file = RESULTS_DIR / f"prevalence_estimation_{lang}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Plot saved: {plot_file}")


# Main Pipeline
def process_language(lang: str) -> Dict:
    """Complete pipeline for one language."""
    lang_name = "English" if lang == "en" else "French"

    print("\n" + "=" * 70)
    print(f"PREVALENCE ESTIMATION - {lang_name.upper()}")
    print("=" * 70)

    # Load features (contains all data including metadata and labels)
    df = load_features(lang)

    print(f"\nDataset: {len(df):,} sentences")
    print(f"  Simple/Vikidia (Label=0):   {(df['Label']==0).sum():,}")
    print(f"  Complex/Wikipedia (Label=1): {(df['Label']==1).sum():,}")

    # Part 1: Naive estimate
    naive_prop = compute_naive_proportion(df)
    print(f"\nNaive estimate (Simple proportion, Label=0/Total): {naive_prop:.4f} ({naive_prop*100:.2f}%)")

    # Part 2: Feature preparation
    features_df, _ = prepare_features(df.copy())
    features_df = handle_outliers(features_df.copy())
    features_df, feature_names = preprocess_features(features_df)
    features_df = remove_high_correlation(features_df)

    # Part 2: Anchor selection
    X_anchor, y_anchor, anchor_idx, anchor_info = select_anchors(df, features_df)

    # Anchor quality summary
    anchor_quality = analyze_anchor_quality(df, anchor_idx, y_anchor)

    # Label-shuffle sanity check on anchors
    shuffle_check = label_shuffle_sanity_check(X_anchor, y_anchor)

    # Part 2: Model training
    best_model, best_name, training_results = train_models(X_anchor, y_anchor)

    # Part 2: Full prediction
    probabilities = predict_full_dataset(best_model, features_df)

    # Part 2: ACC calibration
    tpr, fpr, calibration_info = compute_acc_calibration(best_model, X_anchor, y_anchor)

    # Part 3: Final estimates
    estimates = compute_final_estimates(df, probabilities, tpr, fpr, naive_prop)

    # Additional full-dataset analysis (noise & stratified profiles)
    prediction_analysis = analyze_full_predictions(df, probabilities, lang)

    # Visualization
    plot_results(df, probabilities, estimates, lang)

    # Compile all results
    results = {
        'language': lang_name,
        'dataset_info': {
            'total_sentences': len(df),
            'simple_vikidia_count': int((df['Label'] == 0).sum()),
            'complex_wikipedia_count': int((df['Label'] == 1).sum())
        },
        'anchor_selection': anchor_info,
        'anchor_quality': anchor_quality,
        'anchor_shuffle_sanity_check': shuffle_check,
        'training': {
            'best_model': best_name,
            'model_results': training_results
        },
        'calibration': calibration_info,
        'estimates': estimates,
        'prediction_analysis': prediction_analysis
    }

    # Save results to JSON
    results_file = RESULTS_DIR / f"prevalence_estimation_{lang}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_file}")

    # Cleanup
    del df, features_df, probabilities
    gc.collect()

    return results


def print_final_summary(en_results: Dict, fr_results: Dict) -> None:
    """Print final comparison summary."""
    print("\n" + "=" * 70)
    print("FINAL SUMMARY: ENGLISH vs FRENCH")
    print("=" * 70)

    print(f"\n{'Metric':<45} {'English':<15} {'French':<15}")
    print("-" * 75)

    # Dataset info
    print(f"{'Total sentences':<45} "
          f"{en_results['dataset_info']['total_sentences']:<15,} "
          f"{fr_results['dataset_info']['total_sentences']:<15,}")

    # Naive estimate
    print(f"{'Naive estimate (Label=0/Total)':<45} "
          f"{en_results['estimates']['naive_proportion']:<15.4f} "
          f"{fr_results['estimates']['naive_proportion']:<15.4f}")

    # ACC-corrected estimate
    print(f"{'ACC-corrected true proportion':<45} "
          f"{en_results['estimates']['acc_corrected_proportion']:<15.4f} "
          f"{fr_results['estimates']['acc_corrected_proportion']:<15.4f}")

    # Wikipedia internal
    print(f"{'Wikipedia internal (soft estimate)':<45} "
          f"{en_results['estimates']['wikipedia_analysis']['soft_estimate']:<15.4f} "
          f"{fr_results['estimates']['wikipedia_analysis']['soft_estimate']:<15.4f}")

    print(f"{'Wikipedia internal (hard estimate, P>=' + str(HIGH_CONF_THRESHOLD) + ')':<45} "
          f"{en_results['estimates']['wikipedia_analysis']['hard_estimate']:<15.4f} "
          f"{fr_results['estimates']['wikipedia_analysis']['hard_estimate']:<15.4f}")

    # Calibration info
    print(f"\n{'Calibration (TPR)':<45} "
          f"{en_results['calibration']['tpr']:<15.4f} "
          f"{fr_results['calibration']['tpr']:<15.4f}")
    print(f"{'Calibration (FPR)':<45} "
          f"{en_results['calibration']['fpr']:<15.4f} "
          f"{fr_results['calibration']['fpr']:<15.4f}")

    # Best model
    print(f"\n{'Best model':<45} "
          f"{en_results['training']['best_model']:<15} "
          f"{fr_results['training']['best_model']:<15}")


def main():
    """Main entry point."""
    print("\n" + "=" * 70)
    print("iDEM TASK 1: PREVALENCE ESTIMATION")
    print("Anchor Training + ACC Calibration Pipeline")
    print("=" * 70)
    print(f"Features directory: {FEATURES_DIR}")
    print(f"Results directory:  {RESULTS_DIR}")
    print("=" * 70)

    # Set random seed
    set_seed(RANDOM_SEED)

    # Check local files
    ensure_local_files_exist()

    # Process both languages
    en_results = process_language('en')
    gc.collect()

    fr_results = process_language('fr')
    gc.collect()

    # Final summary
    print_final_summary(en_results, fr_results)

    # Save combined results
    combined = pd.DataFrame([
        {
            'language': 'English',
            'naive_proportion': en_results['estimates']['naive_proportion'],
            'acc_corrected_proportion': en_results['estimates']['acc_corrected_proportion'],
            'wiki_soft_estimate': en_results['estimates']['wikipedia_analysis']['soft_estimate'],
            'wiki_hard_estimate': en_results['estimates']['wikipedia_analysis']['hard_estimate'],
            'tpr': en_results['calibration']['tpr'],
            'fpr': en_results['calibration']['fpr'],
            'best_model': en_results['training']['best_model']
        },
        {
            'language': 'French',
            'naive_proportion': fr_results['estimates']['naive_proportion'],
            'acc_corrected_proportion': fr_results['estimates']['acc_corrected_proportion'],
            'wiki_soft_estimate': fr_results['estimates']['wikipedia_analysis']['soft_estimate'],
            'wiki_hard_estimate': fr_results['estimates']['wikipedia_analysis']['hard_estimate'],
            'tpr': fr_results['calibration']['tpr'],
            'fpr': fr_results['calibration']['fpr'],
            'best_model': fr_results['training']['best_model']
        }
    ])

    results_file = RESULTS_DIR / "prevalence_estimation_summary.csv"
    combined.to_csv(results_file, index=False)
    print(f"\nSummary saved: {results_file}")

    print("\n" + "=" * 70)
    print("TASK 1 COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
