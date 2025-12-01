"""
iDEM Task 2: Complex vs Simple Classifier using Small Transformer
==================================================================

This script fine-tunes DistilBERT to classify sentence complexity
using anchor-based training strategy.

Official Label Definition (from README):
    - Label = 0: Simple (sentence annotated as simple)
    - Label = 1: Complex (sentence annotated as complex)

Source (derived from ID prefix):
    - wiki-* : Wikipedia
    - viki-* : Vikidia

Key Features:
    - Anchor-based training: Clean samples (short Simple + long Complex)
    - Downsampling: Balance class imbalance in anchors
    - ACC Calibration: Adjust for classifier bias
    - Error Analysis: On full corpus to find real misclassifications
    - Full prediction: P(Simple) and P(Complex) for all sentences

Environment: Kaggle Notebook with GPU
Dataset Path: /kaggle/input/dataset-cleaned/

Note: Requires transformers >= 4.46.0 (uses eval_strategy parameter)
"""

# Cell 1: Setup and Imports
import gc
import json
import os
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Hugging Face imports
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

warnings.filterwarnings('ignore')

# Check GPU availability
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# Cell 2: Configuration (English Only)
# Kaggle dataset path
INPUT_DIR = Path("/kaggle/input/dataset-cleaned")
OUTPUT_DIR = Path("/kaggle/working")

# Create output directories
RESULTS_DIR = OUTPUT_DIR / "results"
MODELS_DIR = OUTPUT_DIR / "models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset file (English only)
DATA_FILE = "En-Dataset_cleaned.csv"

# Model configuration
MODEL_CONFIG = {
    "name": "distilbert-base-uncased",
    "max_length": 128,
}

# Training configuration
TRAIN_CONFIG = {
    "batch_size": 32,
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "seed": 42,
    "downsample_ratio": 2.0,  # Majority = N * Minority
    "min_anchor_samples": 100,
}

# Analysis settings
HIGH_CONF_THRESHOLD = 0.9
LOW_CONF_THRESHOLD = 0.1
BOUNDARY_RANGE = (0.45, 0.55)
ERROR_SAMPLE_SIZE = 10


# Cell 3: Reproducibility
def set_seed(seed: int = TRAIN_CONFIG["seed"]) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed()


# Cell 4: Data Loading
def load_data() -> pd.DataFrame:
    """
    Load cleaned dataset.
    
    Official Label Definition:
        - Label = 0: Simple
        - Label = 1: Complex
    
    Source (from ID prefix):
        - wiki-* : Wikipedia
        - viki-* : Vikidia
    """
    filepath = INPUT_DIR / DATA_FILE
    print(f"\nLoading: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Add source column based on ID prefix
    df['source'] = df['ID'].apply(
        lambda x: 'wiki' if str(x).startswith('wiki-') else 'viki'
    )
    
    # Statistics by Label (complexity)
    n_simple = (df['Label'] == 0).sum()
    n_complex = (df['Label'] == 1).sum()
    
    # Statistics by Source
    n_wiki = (df['source'] == 'wiki').sum()
    n_viki = (df['source'] == 'viki').sum()
    
    print(f"\n  Total sentences: {len(df):,}")
    print(f"\n  By Complexity (Label):")
    print(f"    Simple (Label=0):  {n_simple:,} ({n_simple/len(df)*100:.1f}%)")
    print(f"    Complex (Label=1): {n_complex:,} ({n_complex/len(df)*100:.1f}%)")
    print(f"\n  By Source:")
    print(f"    Wikipedia: {n_wiki:,} ({n_wiki/len(df)*100:.1f}%)")
    print(f"    Vikidia:   {n_viki:,} ({n_viki/len(df)*100:.1f}%)")
    
    # Cross-tabulation
    print(f"\n  Cross-tabulation (Source x Label):")
    cross_tab = pd.crosstab(df['source'], df['Label'], margins=True)
    cross_tab.columns = ['Simple(0)', 'Complex(1)', 'Total']
    cross_tab.index = ['Vikidia', 'Wikipedia', 'Total']
    print(cross_tab.to_string())
    
    return df


# Cell 5: Anchor Selection and Downsampling
def select_anchors(
    df: pd.DataFrame,
    downsample_ratio: float = TRAIN_CONFIG["downsample_ratio"],
    min_samples: int = TRAIN_CONFIG["min_anchor_samples"]
) -> Tuple[pd.DataFrame, Dict]:
    """
    Select clean anchor samples for training.
    
    Simple Anchor: Label=0 (Simple) with LengthWords <= Q1
    Complex Anchor: Label=1 (Complex) with LengthWords >= Q3
    
    Downsample majority class to balance.
    """
    print("\n" + "=" * 60)
    print("ANCHOR SELECTION")
    print("=" * 60)
    
    # Calculate quartiles
    q1 = df['LengthWords'].quantile(0.25)
    q3 = df['LengthWords'].quantile(0.75)
    
    print(f"LengthWords quartiles: Q1={q1:.1f}, Q3={q3:.1f}")
    
    # Select anchors based on CORRECT label semantics
    # Label=0 is Simple, Label=1 is Complex
    simple_mask = (df['Label'] == 0) & (df['LengthWords'] <= q1)
    complex_mask = (df['Label'] == 1) & (df['LengthWords'] >= q3)
    
    simple_anchors = df[simple_mask].copy()
    complex_anchors = df[complex_mask].copy()
    
    n_simple = len(simple_anchors)
    n_complex = len(complex_anchors)
    
    # Validation
    if n_simple < min_samples:
        raise ValueError(
            f"Insufficient simple anchors: {n_simple} < {min_samples}"
        )
    if n_complex < min_samples:
        raise ValueError(
            f"Insufficient complex anchors: {n_complex} < {min_samples}"
        )
    
    print(f"\nBefore downsampling:")
    print(f"  Simple anchors (Label=0, LengthWords <= {q1:.1f}): {n_simple:,}")
    print(f"  Complex anchors (Label=1, LengthWords >= {q3:.1f}): {n_complex:,}")
    
    # Determine majority class and downsample
    if n_simple > n_complex:
        minority_count = n_complex
        majority_class = "simple"
        max_majority = int(minority_count * downsample_ratio)
        if n_simple > max_majority:
            simple_anchors = simple_anchors.sample(
                n=max_majority,
                random_state=TRAIN_CONFIG["seed"]
            )
            print(f"\nDownsampling Simple: {n_simple:,} -> {len(simple_anchors):,}")
    else:
        minority_count = n_simple
        majority_class = "complex"
        max_majority = int(minority_count * downsample_ratio)
        if n_complex > max_majority:
            complex_anchors = complex_anchors.sample(
                n=max_majority,
                random_state=TRAIN_CONFIG["seed"]
            )
            print(f"\nDownsampling Complex: {n_complex:,} -> {len(complex_anchors):,}")
    
    # Combine and shuffle
    anchor_df = pd.concat([simple_anchors, complex_anchors], ignore_index=True)
    anchor_df = anchor_df.sample(
        frac=1, random_state=TRAIN_CONFIG["seed"]
    ).reset_index(drop=True)
    
    n_simple_final = (anchor_df['Label'] == 0).sum()
    n_complex_final = (anchor_df['Label'] == 1).sum()
    
    print(f"\nFinal anchor set:")
    print(f"  Total: {len(anchor_df):,}")
    print(f"  Simple (Label=0): {n_simple_final:,}")
    print(f"  Complex (Label=1): {n_complex_final:,}")
    print(f"  Ratio: {max(n_simple_final, n_complex_final) / min(n_simple_final, n_complex_final):.2f}x")
    
    anchor_info = {
        'q1': float(q1),
        'q3': float(q3),
        'n_simple_original': int(n_simple),
        'n_complex_original': int(n_complex),
        'n_simple_final': int(n_simple_final),
        'n_complex_final': int(n_complex_final),
        'majority_class': majority_class,
        'downsample_ratio': downsample_ratio
    }
    
    return anchor_df, anchor_info


def split_anchor_data(
    anchor_df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split anchor data into train/val/test with stratification."""
    print(f"\nSplitting anchors ({train_ratio:.0%}/{val_ratio:.0%}/{test_ratio:.0%})...")
    
    train_df, temp_df = train_test_split(
        anchor_df,
        test_size=(val_ratio + test_ratio),
        stratify=anchor_df['Label'],
        random_state=TRAIN_CONFIG["seed"]
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=temp_df['Label'],
        random_state=TRAIN_CONFIG["seed"]
    )
    
    print(f"  Train: {len(train_df):,} (Simple: {(train_df['Label']==0).sum():,}, Complex: {(train_df['Label']==1).sum():,})")
    print(f"  Val:   {len(val_df):,} (Simple: {(val_df['Label']==0).sum():,}, Complex: {(val_df['Label']==1).sum():,})")
    print(f"  Test:  {len(test_df):,} (Simple: {(test_df['Label']==0).sum():,}, Complex: {(test_df['Label']==1).sum():,})")
    
    return train_df, val_df, test_df


# Cell 6: Tokenization and Dataset Preparation
def prepare_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer,
    max_length: int
) -> Tuple[Dataset, Dataset, Dataset]:
    """Prepare HuggingFace datasets with tokenization."""
    print("\nPreparing datasets...")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["Sentence"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
    
    # Convert to HuggingFace Dataset
    train_ds = Dataset.from_pandas(train_df[['Sentence', 'Label']].reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df[['Sentence', 'Label']].reset_index(drop=True))
    test_ds = Dataset.from_pandas(test_df[['Sentence', 'Label']].reset_index(drop=True))
    
    # Tokenize
    train_ds = train_ds.map(tokenize_function, batched=True, desc="Tokenizing train")
    val_ds = val_ds.map(tokenize_function, batched=True, desc="Tokenizing val")
    test_ds = test_ds.map(tokenize_function, batched=True, desc="Tokenizing test")
    
    # Rename label column for Trainer
    train_ds = train_ds.rename_column("Label", "labels")
    val_ds = val_ds.rename_column("Label", "labels")
    test_ds = test_ds.rename_column("Label", "labels")
    
    # Set format
    columns = ['input_ids', 'attention_mask', 'labels']
    train_ds.set_format(type="torch", columns=columns)
    val_ds.set_format(type="torch", columns=columns)
    test_ds.set_format(type="torch", columns=columns)
    
    print(f"  Train: {len(train_ds):,}, Val: {len(val_ds):,}, Test: {len(test_ds):,}")
    
    return train_ds, val_ds, test_ds


# Cell 7: Custom Trainer with Class Weights
class WeightedTrainer(Trainer):
    """Custom Trainer with class-weighted loss."""
    
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        weights = self.class_weights.to(logits.device)
        loss_fct = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss


def compute_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    """Compute balanced class weights."""
    label_counts = train_df['Label'].value_counts().sort_index()
    n_simple = label_counts[0]  # Label=0 is Simple
    n_complex = label_counts[1]  # Label=1 is Complex
    total = n_simple + n_complex
    
    weight_simple = total / (2 * n_simple)
    weight_complex = total / (2 * n_complex)
    
    weights = torch.tensor([weight_simple, weight_complex], dtype=torch.float32)
    print(f"\nClass weights: [Simple(0): {weight_simple:.4f}, Complex(1): {weight_complex:.4f}]")
    
    return weights


# Cell 8: Metrics Computation
def compute_metrics(eval_pred) -> Dict:
    """
    Compute evaluation metrics.
    
    Label semantics:
        - Label=0: Simple
        - Label=1: Complex (positive class for ROC-AUC)
    """
    logits, labels = eval_pred
    
    # Compute probabilities
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    prob_simple = probs[:, 0]
    prob_complex = probs[:, 1]
    
    # Predictions: Complex (Label=1) is positive class
    preds = (prob_complex >= 0.5).astype(int)
    
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "f1_simple": f1_score(labels, preds, pos_label=0),
        "f1_complex": f1_score(labels, preds, pos_label=1),
        "precision_weighted": precision_score(labels, preds, average="weighted"),
        "recall_weighted": recall_score(labels, preds, average="weighted"),
        "roc_auc": roc_auc_score(labels, prob_complex),  # prob of positive class
    }


# Cell 9: Training Function
def train_model(
    train_ds: Dataset,
    val_ds: Dataset,
    tokenizer,
    class_weights: torch.Tensor
) -> Tuple[Trainer, object]:
    """Train transformer model."""
    print("\n" + "=" * 60)
    print(f"TRAINING: {MODEL_CONFIG['name']}")
    print("=" * 60)
    
    # Load model with CORRECT label mapping
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CONFIG['name'],
        num_labels=2,
        id2label={0: "Simple", 1: "Complex"},
        label2id={"Simple": 0, "Complex": 1}
    )
    
    output_dir = MODELS_DIR / "transformer_en"
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=TRAIN_CONFIG["learning_rate"],
        per_device_train_batch_size=TRAIN_CONFIG["batch_size"],
        per_device_eval_batch_size=TRAIN_CONFIG["batch_size"] * 2,
        num_train_epochs=TRAIN_CONFIG["num_epochs"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
        warmup_ratio=TRAIN_CONFIG["warmup_ratio"],
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        logging_steps=100,
        save_total_limit=2,
        report_to="none",
        seed=TRAIN_CONFIG["seed"],
        fp16=torch.cuda.is_available(),
    )
    
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    
    print("\nStarting training...")
    trainer.train()
    
    return trainer, model


# Cell 10: Evaluation on Test Set
def evaluate_on_test(
    trainer: Trainer,
    test_ds: Dataset,
    test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Evaluate model on test set.
    
    Returns:
        prob_simple, prob_complex, preds, labels, metrics
    """
    print("\n" + "=" * 60)
    print("EVALUATION ON ANCHOR TEST SET")
    print("=" * 60)
    
    pred_output = trainer.predict(test_ds)
    logits = pred_output.predictions
    labels = pred_output.label_ids
    
    # Compute probabilities
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    prob_simple = probs[:, 0]
    prob_complex = probs[:, 1]
    
    # Predictions: pred_label matches official Label semantics
    preds = (prob_complex >= 0.5).astype(int)
    
    # Compute metrics
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted")),
        "f1_simple": float(f1_score(labels, preds, pos_label=0)),
        "f1_complex": float(f1_score(labels, preds, pos_label=1)),
        "precision_simple": float(precision_score(labels, preds, pos_label=0)),
        "precision_complex": float(precision_score(labels, preds, pos_label=1)),
        "recall_simple": float(recall_score(labels, preds, pos_label=0)),
        "recall_complex": float(recall_score(labels, preds, pos_label=1)),
        "roc_auc": float(roc_auc_score(labels, prob_complex)),
        "average_precision": float(average_precision_score(labels, prob_complex)),
    }
    
    print(f"\nTest Metrics:")
    print(f"  Accuracy:            {metrics['accuracy']:.4f}")
    print(f"  F1 (weighted):       {metrics['f1_weighted']:.4f}")
    print(f"  F1 (Simple, Label=0):  {metrics['f1_simple']:.4f}")
    print(f"  F1 (Complex, Label=1): {metrics['f1_complex']:.4f}")
    print(f"  ROC-AUC:             {metrics['roc_auc']:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    print(f"\nConfusion Matrix:")
    print(f"                    Pred Simple(0)  Pred Complex(1)")
    print(f"  Actual Simple(0)  {cm[0,0]:>12,}  {cm[0,1]:>14,}")
    print(f"  Actual Complex(1) {cm[1,0]:>12,}  {cm[1,1]:>14,}")
    
    return prob_simple, prob_complex, preds, labels, metrics


# Cell 11: Full Dataset Prediction
def predict_full_dataset(
    trainer: Trainer,
    full_df: pd.DataFrame,
    tokenizer,
    max_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict probabilities for all sentences.
    
    Returns:
        prob_simple, prob_complex
    """
    print("\n" + "=" * 60)
    print("FULL DATASET PREDICTION")
    print("=" * 60)
    
    print(f"Predicting for {len(full_df):,} sentences...")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["Sentence"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
    
    full_ds = Dataset.from_pandas(full_df[['Sentence']].reset_index(drop=True))
    full_ds = full_ds.map(tokenize_function, batched=True, desc="Tokenizing full")
    full_ds.set_format(type="torch", columns=['input_ids', 'attention_mask'])
    
    pred_output = trainer.predict(full_ds)
    logits = pred_output.predictions
    
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    prob_simple = probs[:, 0]
    prob_complex = probs[:, 1]
    
    print(f"\n  P(Simple) - Mean: {prob_simple.mean():.4f}, Std: {prob_simple.std():.4f}")
    print(f"  P(Complex) - Mean: {prob_complex.mean():.4f}, Std: {prob_complex.std():.4f}")
    print(f"  Predicted Simple (P>=0.5): {(prob_simple >= 0.5).mean():.4f}")
    print(f"  Predicted Complex (P>=0.5): {(prob_complex >= 0.5).mean():.4f}")
    
    return prob_simple, prob_complex


# Cell 12: ACC Calibration
def compute_acc_from_test(
    test_labels: np.ndarray,
    test_preds: np.ndarray
) -> Tuple[float, float, Dict]:
    """
    Compute TPR and FPR from anchor test set for ACC calibration.
    
    Positive class = Complex (Label=1)
    """
    print("\n" + "=" * 60)
    print("ACC CALIBRATION (from anchor test set)")
    print("=" * 60)
    
    cm = confusion_matrix(test_labels, test_preds)
    tn, fp, fn, tp = cm.ravel()
    
    # For Complex as positive class (Label=1):
    # TPR = TP / (TP + FN) = correctly identified Complex
    # FPR = FP / (FP + TN) = Simple misclassified as Complex
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    print(f"\nConfusion Matrix:")
    print(f"  TN (Simple correct)={tn:,}, FP (Simple->Complex)={fp:,}")
    print(f"  FN (Complex->Simple)={fn:,}, TP (Complex correct)={tp:,}")
    print(f"\nRates (Complex as positive):")
    print(f"  TPR (Recall Complex): {tpr:.4f}")
    print(f"  FPR (Fall-out):       {fpr:.4f}")
    print(f"  TNR (Recall Simple):  {tnr:.4f}")
    print(f"  FNR (Miss rate):      {fnr:.4f}")
    
    calibration_info = {
        'method': 'anchor_test_set',
        'n_samples': int(len(test_labels)),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'tpr': float(tpr),
        'fpr': float(fpr),
        'tnr': float(tnr),
        'fnr': float(fnr)
    }
    
    return tpr, fpr, calibration_info


def apply_acc_correction(p_pred: float, tpr: float, fpr: float) -> float:
    """Apply ACC formula: p_true = (p_pred - FPR) / (TPR - FPR)"""
    denominator = tpr - fpr
    if abs(denominator) < 1e-10:
        print("  [WARNING] TPR ~= FPR, using p_pred as fallback")
        return p_pred
    p_true = (p_pred - fpr) / denominator
    return max(0.0, min(1.0, p_true))


# Cell 13: Error Analysis on Full Corpus
def analyze_errors_on_full_corpus(
    full_df: pd.DataFrame,
    prob_simple: np.ndarray,
    prob_complex: np.ndarray,
    sample_size: int = ERROR_SAMPLE_SIZE
) -> Dict:
    """
    Analyze prediction errors on FULL corpus.
    
    This finds real misclassifications, unlike anchor test set
    which may have 100% accuracy due to clean separation.
    
    Label semantics:
        - Label=0: Simple
        - Label=1: Complex
        - pred_label = (prob_complex >= 0.5)
    """
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS ON FULL CORPUS")
    print("=" * 60)
    
    # Build analysis dataframe
    analysis_df = full_df.copy()
    analysis_df['prob_simple'] = prob_simple
    analysis_df['prob_complex'] = prob_complex
    analysis_df['pred_label'] = (prob_complex >= 0.5).astype(int)
    analysis_df['correct'] = analysis_df['Label'] == analysis_df['pred_label']
    
    errors_df = analysis_df[~analysis_df['correct']]
    
    n_total = len(analysis_df)
    n_errors = len(errors_df)
    error_rate = n_errors / n_total if n_total > 0 else 0
    
    print(f"\nTotal errors: {n_errors:,} / {n_total:,} ({error_rate:.2%})")
    
    results = {
        'total_samples': n_total,
        'total_errors': n_errors,
        'error_rate': float(error_rate),
        'false_positives': [],  # Pred=Complex but True=Simple
        'false_negatives': [],  # Pred=Simple but True=Complex
        'boundary_cases': []
    }
    
    # False Positives: pred_label=1 (Complex) but Label=0 (Simple)
    # Simple sentences misclassified as Complex
    fp_df = errors_df[(errors_df['pred_label'] == 1) & (errors_df['Label'] == 0)]
    fp_high_conf = fp_df[fp_df['prob_complex'] >= HIGH_CONF_THRESHOLD]
    fp_sample = fp_df.sort_values('prob_complex', ascending=False).head(sample_size)
    
    print(f"\n[1] FALSE POSITIVES (pred=Complex, true=Simple)")
    print(f"    Total FP: {len(fp_df):,}")
    print(f"    High confidence FP (P(Complex) >= {HIGH_CONF_THRESHOLD}): {len(fp_high_conf):,}")
    print(f"    These Simple sentences were predicted as Complex")
    
    # By source breakdown
    fp_wiki = len(fp_df[fp_df['source'] == 'wiki'])
    fp_viki = len(fp_df[fp_df['source'] == 'viki'])
    print(f"    By source: Wikipedia={fp_wiki:,}, Vikidia={fp_viki:,}")
    
    for i, (_, row) in enumerate(fp_sample.iterrows(), 1):
        sample = {
            'sentence': row['Sentence'][:200],
            'source': row['source'],
            'true_label': int(row['Label']),
            'pred_label': int(row['pred_label']),
            'prob_simple': float(row['prob_simple']),
            'prob_complex': float(row['prob_complex']),
            'length_words': int(row['LengthWords']) if 'LengthWords' in row else None
        }
        results['false_positives'].append(sample)
        print(f"\n    FP-{i} [P(Complex)={row['prob_complex']:.3f}] [Source={row['source']}] [Len={row.get('LengthWords', 'N/A')}]:")
        print(f"    {row['Sentence'][:150]}...")
    
    # False Negatives: pred_label=0 (Simple) but Label=1 (Complex)
    # Complex sentences misclassified as Simple
    fn_df = errors_df[(errors_df['pred_label'] == 0) & (errors_df['Label'] == 1)]
    fn_high_conf = fn_df[fn_df['prob_simple'] >= HIGH_CONF_THRESHOLD]
    fn_sample = fn_df.sort_values('prob_simple', ascending=False).head(sample_size)
    
    print(f"\n[2] FALSE NEGATIVES (pred=Simple, true=Complex)")
    print(f"    Total FN: {len(fn_df):,}")
    print(f"    High confidence FN (P(Simple) >= {HIGH_CONF_THRESHOLD}): {len(fn_high_conf):,}")
    print(f"    These Complex sentences were predicted as Simple")
    
    fn_wiki = len(fn_df[fn_df['source'] == 'wiki'])
    fn_viki = len(fn_df[fn_df['source'] == 'viki'])
    print(f"    By source: Wikipedia={fn_wiki:,}, Vikidia={fn_viki:,}")
    
    for i, (_, row) in enumerate(fn_sample.iterrows(), 1):
        sample = {
            'sentence': row['Sentence'][:200],
            'source': row['source'],
            'true_label': int(row['Label']),
            'pred_label': int(row['pred_label']),
            'prob_simple': float(row['prob_simple']),
            'prob_complex': float(row['prob_complex']),
            'length_words': int(row['LengthWords']) if 'LengthWords' in row else None
        }
        results['false_negatives'].append(sample)
        print(f"\n    FN-{i} [P(Simple)={row['prob_simple']:.3f}] [Source={row['source']}] [Len={row.get('LengthWords', 'N/A')}]:")
        print(f"    {row['Sentence'][:150]}...")
    
    # Boundary cases: P close to 0.5
    boundary_df = analysis_df[
        (analysis_df['prob_simple'] >= BOUNDARY_RANGE[0]) &
        (analysis_df['prob_simple'] <= BOUNDARY_RANGE[1])
    ].copy()
    
    if len(boundary_df) > 0:
        boundary_df['dist_from_05'] = (boundary_df['prob_simple'] - 0.5).abs()
        boundary_sample = boundary_df.nsmallest(sample_size, 'dist_from_05')
    else:
        boundary_sample = boundary_df
    
    print(f"\n[3] BOUNDARY CASES (P(Simple) in {BOUNDARY_RANGE})")
    print(f"    Total boundary: {len(boundary_df):,}")
    
    for i, (_, row) in enumerate(boundary_sample.iterrows(), 1):
        sample = {
            'sentence': row['Sentence'][:200],
            'source': row['source'],
            'true_label': int(row['Label']),
            'pred_label': int(row['pred_label']),
            'prob_simple': float(row['prob_simple']),
            'correct': bool(row['correct'])
        }
        results['boundary_cases'].append(sample)
        status = "CORRECT" if row['correct'] else "WRONG"
        label_name = "Simple" if row['Label'] == 0 else "Complex"
        print(f"\n    B-{i} [P(Simple)={row['prob_simple']:.3f}] [{status}] [True={label_name}]:")
        print(f"    {row['Sentence'][:150]}...")
    
    # Summary
    results['summary'] = {
        'total_fp': len(fp_df),
        'total_fn': len(fn_df),
        'high_conf_fp': len(fp_high_conf),
        'high_conf_fn': len(fn_high_conf),
        'total_boundary': len(boundary_df),
        'fp_by_source': {'wiki': fp_wiki, 'viki': fp_viki},
        'fn_by_source': {'wiki': fn_wiki, 'viki': fn_viki}
    }
    
    return results


# Cell 14: Final Estimates
def compute_final_estimates(
    full_df: pd.DataFrame,
    prob_simple: np.ndarray,
    prob_complex: np.ndarray,
    tpr: float,
    fpr: float
) -> Dict:
    """
    Compute final prevalence estimates with ACC correction.
    
    Label semantics:
        - Label=0: Simple
        - Label=1: Complex (positive class for ACC)
    """
    print("\n" + "=" * 60)
    print("FINAL ESTIMATES")
    print("=" * 60)
    
    # Naive estimate from labels
    naive_simple = (full_df['Label'] == 0).mean()
    naive_complex = (full_df['Label'] == 1).mean()
    
    # Predicted proportions
    pred_simple = (prob_simple >= 0.5).mean()
    pred_complex = (prob_complex >= 0.5).mean()
    
    # ACC correction for Complex proportion
    # p_true_complex = (p_pred_complex - FPR) / (TPR - FPR)
    acc_complex = apply_acc_correction(pred_complex, tpr, fpr)
    acc_simple = 1.0 - acc_complex
    
    print(f"\n1. GLOBAL SENTENCE COMPLEXITY:")
    print(f"   Naive (from Label):")
    print(f"     Simple (Label=0): {naive_simple:.4f} ({naive_simple*100:.2f}%)")
    print(f"     Complex (Label=1): {naive_complex:.4f} ({naive_complex*100:.2f}%)")
    print(f"   Predicted (P >= 0.5):")
    print(f"     Simple: {pred_simple:.4f} ({pred_simple*100:.2f}%)")
    print(f"     Complex: {pred_complex:.4f} ({pred_complex*100:.2f}%)")
    print(f"   ACC-corrected:")
    print(f"     Simple: {acc_simple:.4f} ({acc_simple*100:.2f}%)")
    print(f"     Complex: {acc_complex:.4f} ({acc_complex*100:.2f}%)")
    
    # Wikipedia internal analysis
    wiki_mask = full_df['source'] == 'wiki'
    wiki_prob_simple = prob_simple[wiki_mask]
    wiki_prob_complex = prob_complex[wiki_mask]
    n_wiki = wiki_mask.sum()
    
    wiki_pred_simple = (wiki_prob_simple >= 0.5).mean()
    wiki_high_simple = (wiki_prob_simple >= HIGH_CONF_THRESHOLD).mean()
    
    print(f"\n2. WIKIPEDIA INTERNAL (Source=wiki):")
    print(f"   Total sentences: {n_wiki:,}")
    print(f"   Soft estimate (mean P(Simple)): {wiki_prob_simple.mean():.4f}")
    print(f"   Predicted Simple (P >= 0.5): {wiki_pred_simple:.4f} ({wiki_pred_simple*100:.2f}%)")
    print(f"   High conf Simple (P >= {HIGH_CONF_THRESHOLD}): {wiki_high_simple:.4f} ({wiki_high_simple*100:.2f}%)")
    
    # Vikidia internal analysis
    viki_mask = full_df['source'] == 'viki'
    viki_prob_simple = prob_simple[viki_mask]
    viki_prob_complex = prob_complex[viki_mask]
    n_viki = viki_mask.sum()
    
    viki_pred_complex = (viki_prob_complex >= 0.5).mean()
    viki_high_complex = (viki_prob_complex >= HIGH_CONF_THRESHOLD).mean()
    
    print(f"\n3. VIKIDIA INTERNAL (Source=viki):")
    print(f"   Total sentences: {n_viki:,}")
    print(f"   Soft estimate (mean P(Complex)): {viki_prob_complex.mean():.4f}")
    print(f"   Predicted Complex (P >= 0.5): {viki_pred_complex:.4f} ({viki_pred_complex*100:.2f}%)")
    print(f"   High conf Complex (P >= {HIGH_CONF_THRESHOLD}): {viki_high_complex:.4f} ({viki_high_complex*100:.2f}%)")
    
    estimates = {
        'naive': {
            'simple': float(naive_simple),
            'complex': float(naive_complex)
        },
        'predicted': {
            'simple': float(pred_simple),
            'complex': float(pred_complex)
        },
        'acc_corrected': {
            'simple': float(acc_simple),
            'complex': float(acc_complex)
        },
        'wikipedia_analysis': {
            'n_sentences': int(n_wiki),
            'mean_prob_simple': float(wiki_prob_simple.mean()),
            'pred_simple_rate': float(wiki_pred_simple),
            'high_conf_simple_rate': float(wiki_high_simple)
        },
        'vikidia_analysis': {
            'n_sentences': int(n_viki),
            'mean_prob_complex': float(viki_prob_complex.mean()),
            'pred_complex_rate': float(viki_pred_complex),
            'high_conf_complex_rate': float(viki_high_complex)
        }
    }
    
    return estimates


# Cell 15: Main Pipeline
def main() -> Dict:
    """Main execution pipeline for English only."""
    print("\n" + "=" * 70)
    print("iDEM TASK 2: TRANSFORMER FINE-TUNING (English)")
    print("Anchor-based Training + ACC Calibration")
    print("=" * 70)
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Model:  {MODEL_CONFIG['name']}")
    print(f"Seed:   {TRAIN_CONFIG['seed']}")
    print("=" * 70)
    
    set_seed()
    
    # Load data
    full_df = load_data()
    
    # Select and split anchors
    anchor_df, anchor_info = select_anchors(full_df)
    train_df, val_df, test_df = split_anchor_data(anchor_df)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG['name'])
    
    # Prepare datasets
    train_ds, val_ds, test_ds = prepare_datasets(
        train_df, val_df, test_df,
        tokenizer, MODEL_CONFIG['max_length']
    )
    
    # Compute class weights
    class_weights = compute_class_weights(train_df)
    
    # Train model
    trainer, model = train_model(train_ds, val_ds, tokenizer, class_weights)
    
    # Evaluate on anchor test set
    prob_simple_test, prob_complex_test, preds_test, labels_test, test_metrics = \
        evaluate_on_test(trainer, test_ds, test_df)
    
    # ACC calibration
    tpr, fpr, calibration_info = compute_acc_from_test(labels_test, preds_test)
    
    # Full dataset prediction
    prob_simple_full, prob_complex_full = predict_full_dataset(
        trainer, full_df, tokenizer, MODEL_CONFIG['max_length']
    )
    
    # Error analysis on FULL corpus (not anchor test set)
    error_analysis = analyze_errors_on_full_corpus(
        full_df, prob_simple_full, prob_complex_full
    )
    
    # Final estimates
    estimates = compute_final_estimates(
        full_df, prob_simple_full, prob_complex_full, tpr, fpr
    )
    
    # Save predictions
    pred_df = full_df[['ID', 'Sentence', 'Label', 'source', 'LengthWords']].copy()
    pred_df['prob_simple'] = prob_simple_full
    pred_df['prob_complex'] = prob_complex_full
    pred_df['pred_label'] = (prob_complex_full >= 0.5).astype(int)
    pred_df.to_csv(RESULTS_DIR / "transformer_predictions_en.csv", index=False)
    print(f"\nPredictions saved: {RESULTS_DIR / 'transformer_predictions_en.csv'}")
    
    # Compile results
    results = {
        'model': MODEL_CONFIG['name'],
        'dataset_info': {
            'total_sentences': len(full_df),
            'simple_count': int((full_df['Label'] == 0).sum()),
            'complex_count': int((full_df['Label'] == 1).sum()),
            'wikipedia_count': int((full_df['source'] == 'wiki').sum()),
            'vikidia_count': int((full_df['source'] == 'viki').sum())
        },
        'anchor_info': anchor_info,
        'test_metrics': test_metrics,
        'calibration': calibration_info,
        'estimates': estimates,
        'error_analysis': error_analysis
    }
    
    # Save results JSON
    results_file = RESULTS_DIR / "transformer_results_en.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    print(f"Results saved: {results_file}")
    
    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\nAnchor Test Metrics:")
    print(f"  Accuracy:     {test_metrics['accuracy']:.4f}")
    print(f"  F1 (weighted): {test_metrics['f1_weighted']:.4f}")
    print(f"  ROC-AUC:      {test_metrics['roc_auc']:.4f}")
    print(f"\nACC Calibration:")
    print(f"  TPR: {calibration_info['tpr']:.4f}")
    print(f"  FPR: {calibration_info['fpr']:.4f}")
    print(f"\nPrevalence Estimates:")
    print(f"  Naive Simple:       {estimates['naive']['simple']:.4f}")
    print(f"  ACC-corrected Simple: {estimates['acc_corrected']['simple']:.4f}")
    print(f"\nError Analysis (Full Corpus):")
    print(f"  Total errors: {error_analysis['total_errors']:,} ({error_analysis['error_rate']:.2%})")
    print(f"  FP (Simple->Complex): {error_analysis['summary']['total_fp']:,}")
    print(f"  FN (Complex->Simple): {error_analysis['summary']['total_fn']:,}")
    
    print("\n" + "=" * 70)
    print("TASK 2 COMPLETE!")
    print("=" * 70)
    
    # Cleanup
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    
    return results


# Cell 16: Run
if __name__ == "__main__":
    results = main()