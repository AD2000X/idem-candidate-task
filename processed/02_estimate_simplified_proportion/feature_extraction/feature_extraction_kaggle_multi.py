"""
iDEM Feature Engineering - Kaggle Notebook Version (T4 GPU + 4 CPU)
====================================================================
Complete feature extraction pipeline for text complexity classification.
Optimized for Kaggle Notebook with T4 GPU and 4 CPU cores.

** WITH CHECKPOINT SUPPORT **
If disconnected, re-run and it will resume from last checkpoint.

Hardware Utilization:
    - 4 CPU cores: Basic feature extraction (via joblib)
    - T4 GPU: Sentence-BERT embeddings (single GPU, large batch size)
    - Single-thread: spaCy parsing (Notebook compatibility mode)

Note: Multi-process encoding for spaCy and SBERT is disabled due to 
Kaggle Notebook (Jupyter/IPython) multiprocessing limitations.

Pipeline Steps:
    1. Load raw En-Dataset.csv / Fr-Dataset.csv
    2. Assign unique Index: viki-000001, wiki-000001...
    3. Basic cleaning: remove NaN, blank rows
    4. Duplicate detection and removal:
       - 4a: Vikidia internal duplicates
       - 4b: Wikipedia internal duplicates
       - 4c: Cross-dataset duplicates (Leakage) - remove from Vikidia
    5. Save En-Dataset_cleaned.csv / Fr-Dataset_cleaned.csv
    6. Extract ALL features for ENTIRE dataset
    7. Compute Sentence-BERT embeddings + Cosine Similarity (GPU)
    8. Output en_cleaned_features.csv / fr_cleaned_features.csv

Checkpoints saved:
    - checkpoint_{lang}_cleaned.csv      (after Step 4)
    - checkpoint_{lang}_basic.csv        (after basic features)
    - checkpoint_{lang}_spacy.csv        (after spaCy features)
    - checkpoint_{lang}_sbert.npy        (after SBERT embeddings)

Output Columns (26 total):
    - Index columns: Index, ID, Name, Sentence, Label, LengthWords, LengthChars
    - New features: words_chars_ratio, cos_simi
    - Basic features: avg_word_len, long_word_ratio, ttr, punct_density,
                      comma_density, digit_ratio, upper_ratio, has_parens
    - spaCy features: n_tokens, max_depth, avg_depth, avg_dependency_distance,
                      func_word_ratio, n_clauses, clause_ratio, noun_ratio, verb_ratio

Kaggle Setup:
    1. Enable GPU: Settings > Accelerator > GPU T4 x2
    2. Enable Internet: Settings > Internet > On
    3. Upload En-Dataset.csv and Fr-Dataset.csv to /kaggle/input/

Run in Kaggle Notebook:
    # Cell 1: Install dependencies
    !pip install sentence-transformers spacy joblib -q
    !python -m spacy download en_core_web_sm -q
    !python -m spacy download fr_core_news_sm -q
    
    # Cell 2: Run pipeline
    %run feature_extraction_kaggle_multi.py
"""


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import warnings
import multiprocessing
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')


# Hardware Configuration
def get_hardware_config() -> Dict: 
    config = {
        'n_cpus': multiprocessing.cpu_count(),
        'n_gpus': 0,
        'gpu_names': [],
        'gpu_memory': [],
        'device': 'cpu',
        'use_multi_gpu': False  # Disabled for Kaggle Notebook compatibility
    }
    
    if torch.cuda.is_available():
        config['n_gpus'] = torch.cuda.device_count()
        config['device'] = 'cuda'
        
        for i in range(config['n_gpus']):
            props = torch.cuda.get_device_properties(i)
            config['gpu_names'].append(props.name)
            config['gpu_memory'].append(props.total_memory / 1e9)
        
        # NOTE: Multi-GPU is disabled due to Kaggle Notebook limitations
        # config['use_multi_gpu'] = False (keep as False)
    
    return config


def print_hardware_info(config: Dict) -> None:
    print("=" * 70)
    print("HARDWARE CONFIGURATION")
    print("=" * 70)
    
    print(f"\nCPU:")
    print(f"  Available cores: {config['n_cpus']}")
    print(f"  Using for basic features: {min(config['n_cpus'], 4)} (joblib)")
    print(f"  Using for spaCy: 1 (single-thread for Notebook compatibility)")
    
    print(f"\nGPU:")
    if config['n_gpus'] > 0:
        print(f"  Available GPUs: {config['n_gpus']}")
        for i in range(config['n_gpus']):
            print(f"    GPU {i}: {config['gpu_names'][i]} ({config['gpu_memory'][i]:.1f} GB)")
        print(f"  Using: Single GPU (multi-GPU disabled for Notebook compatibility)")
        print(f"  Primary device: {config['device']}")
    else:
        print("  No GPU available (using CPU)")
    
    print("=" * 70)


def check_environment():
    print("=" * 70)
    print("ENVIRONMENT CHECK")
    print("=" * 70)
    
    # Get hardware config
    hw_config = get_hardware_config()
    
    # Print basic info
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
    
    is_kaggle = os.path.exists('/kaggle')
    print(f"Kaggle Environment: {is_kaggle}")
    print(f"TOKENIZERS_PARALLELISM: {os.environ.get('TOKENIZERS_PARALLELISM', 'not set')}")
    
    print("=" * 70)
    
    return hw_config, is_kaggle


# Configuration
def setup_paths(is_kaggle: bool) -> Tuple[Path, Path, Path]:
    if is_kaggle:
        INPUT_DIR = Path('/kaggle/input')
        WORKING_DIR = Path('/kaggle/working')
        
        data_dirs = list(INPUT_DIR.glob('*'))
        if data_dirs:
            DATA_DIR = data_dirs[0]
        else:
            DATA_DIR = INPUT_DIR
        
        OUTPUT_DIR = WORKING_DIR / 'output'
        CHECKPOINT_DIR = WORKING_DIR / 'checkpoints'
    else:
        BASE_DIR = Path('.')
        DATA_DIR = BASE_DIR / 'data'
        OUTPUT_DIR = BASE_DIR / 'output'
        CHECKPOINT_DIR = BASE_DIR / 'checkpoints'
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Checkpoint directory: {CHECKPOINT_DIR}")
    
    return DATA_DIR, OUTPUT_DIR, CHECKPOINT_DIR


# Batch processing config
BATCH_SIZE = 5000
SBERT_BATCH_SIZE = 128  # Base batch size (will be doubled for T4)

RANDOM_SEED = 42

# Feature names
BASIC_FEATURE_NAMES = [
    'avg_word_len',
    'long_word_ratio',
    'ttr',
    'punct_density',
    'comma_density',
    'digit_ratio',
    'upper_ratio',
    'has_parens'
]

SPACY_FEATURE_NAMES = [
    'n_tokens',
    'max_depth',
    'avg_depth',
    'avg_dependency_distance',
    'func_word_ratio',
    'n_clauses',
    'clause_ratio',
    'noun_ratio',
    'verb_ratio'
]

FUNCTION_WORD_POS = {'DET', 'ADP', 'PRON', 'CCONJ', 'SCONJ', 'AUX', 'PART'}
CLAUSE_DEPS = {'advcl', 'ccomp', 'acl', 'relcl', 'xcomp'}


# ============================================================
# Checkpoint Functions
# ============================================================

def get_checkpoint_path(checkpoint_dir: Path, lang: str, step: str, ext: str = 'csv') -> Path:
    return checkpoint_dir / f"checkpoint_{lang}_{step}.{ext}"


def checkpoint_exists(checkpoint_dir: Path, lang: str, step: str, ext: str = 'csv') -> bool:
    return get_checkpoint_path(checkpoint_dir, lang, step, ext).exists()


def save_checkpoint_csv(df: pd.DataFrame, checkpoint_dir: Path, lang: str, step: str) -> None:
    path = get_checkpoint_path(checkpoint_dir, lang, step, 'csv')
    df.to_csv(path, index=False)
    print(f"  [Checkpoint saved] {path.name}")


def load_checkpoint_csv(checkpoint_dir: Path, lang: str, step: str) -> pd.DataFrame:
    path = get_checkpoint_path(checkpoint_dir, lang, step, 'csv')
    print(f"  [Checkpoint loaded] {path.name}")
    return pd.read_csv(path)


def save_checkpoint_npy(arr: np.ndarray, checkpoint_dir: Path, lang: str, step: str) -> None:
    path = get_checkpoint_path(checkpoint_dir, lang, step, 'npy')
    np.save(path, arr)
    print(f"  [Checkpoint saved] {path.name}")


def load_checkpoint_npy(checkpoint_dir: Path, lang: str, step: str) -> np.ndarray:
    path = get_checkpoint_path(checkpoint_dir, lang, step, 'npy')
    print(f"  [Checkpoint loaded] {path.name}")
    return np.load(path)


def clear_checkpoints(checkpoint_dir: Path, lang: str) -> None:
    for f in checkpoint_dir.glob(f"checkpoint_{lang}_*"):
        f.unlink()
        print(f"  [Checkpoint deleted] {f.name}")


# Step 1: Load Data
def load_data(lang: str, data_dir: Path) -> pd.DataFrame:
    filename = "En-Dataset.csv" if lang == 'en' else "Fr-Dataset.csv"
    
    print(f"\nStep 1: Loading data")
    
    # Try multiple possible paths
    possible_paths = [
        data_dir / "data" / filename,
        data_dir / filename,
        data_dir / filename.lower(),
        data_dir / "data" / filename.lower(),
    ]
    
    input_file = None
    for path in possible_paths:
        if path.exists():
            input_file = path
            break
    
    if input_file is None:
        print(f"  Searching for {filename}...")
        found_files = list(data_dir.rglob(f"*{filename}")) + list(data_dir.rglob(f"*{filename.lower()}"))
        if found_files:
            input_file = found_files[0]
        else:
            raise FileNotFoundError(
                f"Dataset not found. Searched in:\n"
                f"  {data_dir}\n"
                f"  Tried: {[str(p) for p in possible_paths]}"
            )
    
    print(f"  File: {input_file}")
    
    df = pd.read_csv(input_file)
    print(f"  Loaded {len(df):,} rows")
    
    return df


# Step 2: Assign Unique Index
def assign_unique_index(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\nStep 2: Assigning unique Index")
    
    df['source'] = df['ID'].apply(
        lambda x: 'viki' if str(x).lower().startswith('viki') else 'wiki'
    )
    
    viki_mask = df['source'] == 'viki'
    wiki_mask = df['source'] == 'wiki'
    
    viki_indices = [f"viki-{i:06d}" for i in range(1, viki_mask.sum() + 1)]
    wiki_indices = [f"wiki-{i:06d}" for i in range(1, wiki_mask.sum() + 1)]
    
    df.loc[viki_mask, 'Index'] = viki_indices
    df.loc[wiki_mask, 'Index'] = wiki_indices
    
    print(f"  Vikidia sentences: {viki_mask.sum():,}")
    print(f"  Wikipedia sentences: {wiki_mask.sum():,}")
    
    return df


# Step 3: Basic Cleaning
def basic_cleaning(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    print(f"\nStep 3: Basic cleaning")
    
    stats = {'original_rows': len(df)}
    
    critical_cols = ['Index', 'Sentence', 'Label']
    nan_before = len(df)
    df = df.dropna(subset=critical_cols)
    stats['nan_removed'] = nan_before - len(df)
    print(f"  NaN removed: {stats['nan_removed']:,}")
    
    blank_before = len(df)
    df = df[df['Sentence'].str.strip().str.len() > 0]
    stats['blank_removed'] = blank_before - len(df)
    print(f"  Blank removed: {stats['blank_removed']:,}")
    
    return df, stats


# Step 4: Duplicate Detection and Removal
def remove_duplicates(df: pd.DataFrame, stats: Dict) -> Tuple[pd.DataFrame, Dict]:
    print(f"\nStep 4: Duplicate detection and removal")
    
    viki_df = df[df['source'] == 'viki'].copy()
    wiki_df = df[df['source'] == 'wiki'].copy()
    
    print(f"  Before cleaning:")
    print(f"    Vikidia: {len(viki_df):,}")
    print(f"    Wikipedia: {len(wiki_df):,}")
    
    viki_before = len(viki_df)
    viki_df = viki_df.drop_duplicates(subset=['Sentence'], keep='first')
    stats['viki_internal_dup'] = viki_before - len(viki_df)
    print(f"  4a. Vikidia internal duplicates: {stats['viki_internal_dup']:,}")
    
    wiki_before = len(wiki_df)
    wiki_df = wiki_df.drop_duplicates(subset=['Sentence'], keep='first')
    stats['wiki_internal_dup'] = wiki_before - len(wiki_df)
    print(f"  4b. Wikipedia internal duplicates: {stats['wiki_internal_dup']:,}")
    
    wiki_sentences = set(wiki_df['Sentence'].values)
    leakage_mask = viki_df['Sentence'].isin(wiki_sentences)
    stats['leakage_removed'] = leakage_mask.sum()
    viki_df = viki_df[~leakage_mask]
    print(f"  4c. Leakage (Viki ∩ Wiki): {stats['leakage_removed']:,}")
    
    df = pd.concat([viki_df, wiki_df], ignore_index=True)
    df = df.sort_values('Index').reset_index(drop=True)
    
    print(f"  After cleaning: {len(df):,}")
    
    stats['after_dedup'] = len(df)
    stats['viki_final'] = len(viki_df)
    stats['wiki_final'] = len(wiki_df)
    
    return df, stats


# Step 5: Save Cleaned Dataset
def save_cleaned_dataset(df: pd.DataFrame, lang: str, output_dir: Path) -> Path:
    print(f"\nStep 5: Saving cleaned dataset")
    
    cols_to_save = ['Index', 'ID', 'Name', 'Sentence', 'Label', 'LengthWords', 'LengthChars', 'source']
    df_to_save = df[cols_to_save]
    
    output_file = output_dir / f"{lang.capitalize()}-Dataset_cleaned.csv"
    
    df_to_save.to_csv(output_file, index=False)
    print(f"  Saved: {output_file}")
    print(f"  Rows: {len(df_to_save):,}")
    
    return output_file


# Step 6: Feature Extraction
def extract_basic_features_single(sent: str) -> List[float]:
    words = sent.split()
    chars = list(sent)
    
    if len(words) == 0:
        return [0.0] * 8
    
    avg_word_len = np.mean([len(w) for w in words])
    long_word_ratio = sum(1 for w in words if len(w) > 6) / len(words)
    
    words_norm = words[:50] if len(words) > 50 else words
    ttr = len(set(w.lower() for w in words_norm)) / len(words_norm)
    
    punct_count = sum(1 for c in chars if c in '.,;:!?()[]{}"-')
    punct_density = punct_count / len(words)
    
    comma_density = sent.count(',') / len(words)
    digit_ratio = sum(1 for c in chars if c.isdigit()) / len(chars) if len(chars) > 0 else 0
    upper_ratio = sum(1 for c in chars if c.isupper()) / len(chars) if len(chars) > 0 else 0
    has_parens = 1.0 if '(' in sent else 0.0
    
    return [
        avg_word_len, long_word_ratio, ttr, punct_density,
        comma_density, digit_ratio, upper_ratio, has_parens
    ]


def extract_basic_features(sentences: List[str], n_jobs: int = 4) -> pd.DataFrame:
    print(f"    Using {n_jobs} CPU cores for basic features (joblib)...")
    
    # Use joblib for parallel processing
    features = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(extract_basic_features_single)(sent) 
        for sent in tqdm(sentences, desc="Basic features", unit="sent")
    )
    
    return pd.DataFrame(features, columns=BASIC_FEATURE_NAMES, dtype=np.float32)


def get_dependency_depth(token) -> int:
    depth = 0
    current = token
    while current.head != current:
        depth += 1
        current = current.head
    return depth


def extract_spacy_features_single(doc) -> List[float]:
    tokens = [token for token in doc if not token.is_space]
    n_tokens = len(tokens)
    
    if n_tokens == 0:
        return [0.0] * 9
    
    depths = [get_dependency_depth(token) for token in tokens]
    max_depth = max(depths) if depths else 0
    avg_depth = np.mean(depths) if depths else 0.0
    
    dep_distances = [abs(token.i - token.head.i) for token in tokens]
    avg_dependency_distance = np.mean(dep_distances) if dep_distances else 0.0
    
    func_word_count = sum(1 for token in tokens if token.pos_ in FUNCTION_WORD_POS)
    func_word_ratio = func_word_count / n_tokens
    
    n_clauses = sum(1 for token in tokens if token.dep_ in CLAUSE_DEPS)
    clause_ratio = n_clauses / n_tokens
    
    noun_count = sum(1 for token in tokens if token.pos_ in {'NOUN', 'PROPN'})
    verb_count = sum(1 for token in tokens if token.pos_ == 'VERB')
    noun_ratio = noun_count / n_tokens
    verb_ratio = verb_count / n_tokens
    
    return [
        float(n_tokens), float(max_depth), float(avg_depth),
        float(avg_dependency_distance), float(func_word_ratio),
        float(n_clauses), float(clause_ratio),
        float(noun_ratio), float(verb_ratio)
    ]


def load_spacy_model(lang: str):
    import spacy
    
    model_name = 'en_core_web_sm' if lang == 'en' else 'fr_core_news_sm'
    print(f"  Loading spaCy model: {model_name}")
    
    try:
        nlp = spacy.load(model_name, disable=['ner', 'textcat'])
    except OSError:
        print(f"  Downloading {model_name}...")
        os.system(f"python -m spacy download {model_name}")
        nlp = spacy.load(model_name, disable=['ner', 'textcat'])
    
    nlp.max_length = 100000
    
    return nlp


def extract_spacy_features(sentences: List[str], nlp) -> pd.DataFrame:
    print(f"    Processing with single-thread (Notebook compatibility mode)...")
    
    all_features = []
    
    for doc in tqdm(
        nlp.pipe(sentences, batch_size=1000), 
        total=len(sentences), 
        desc="spaCy features", 
        unit="sent"
    ):
        features = extract_spacy_features_single(doc)
        all_features.append(features)
        del doc
    
    return pd.DataFrame(all_features, columns=SPACY_FEATURE_NAMES, dtype=np.float32)


# Step 7: Sentence-BERT Embeddings + Cosine Similarity (GPU)
def compute_sbert_embeddings(
    sentences: List[str],
    lang: str,
    hw_config: Dict,
    batch_size: int = SBERT_BATCH_SIZE
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    
    print("\n  Computing Sentence-BERT embeddings...")
    print(f"    Device: {hw_config['device']}")
    print(f"    GPUs available: {hw_config['n_gpus']} (using single GPU)")
    
    if lang == 'en':
        model_name = 'all-MiniLM-L6-v2'
    else:
        model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
    
    print(f"    Model: {model_name}")
    
    # Load model on primary GPU
    model = SentenceTransformer(model_name, device=hw_config['device'])
    
    if hw_config['n_gpus'] > 0 and hw_config['device'] == 'cuda':
        # Use larger batch size for faster processing on T4
        effective_batch_size = batch_size * 2  # 256 for T4
        print(f"    Using enlarged batch size for T4: {effective_batch_size}")
    else:
        effective_batch_size = batch_size
    
    print(f"    Total sentences: {len(sentences):,}")
    print(f"    Batch size: {effective_batch_size}")
    
    start_time = time.time()
    embeddings = model.encode(
        sentences,
        batch_size=effective_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    encode_time = time.time() - start_time
    print(f"    Encoding time: {encode_time:.1f}s ({len(sentences)/encode_time:.0f} sent/s)")
    
    del model
    if hw_config['device'] == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    return embeddings


def compute_cosine_similarity_from_embeddings(
    embeddings: np.ndarray,
    df: pd.DataFrame
) -> np.ndarray:
    """Compute cosine similarity from pre-computed embeddings."""
    from sklearn.metrics.pairwise import cosine_similarity
    
    print("\n  Computing cosine similarity from embeddings...")
    
    viki_mask = df['source'] == 'viki'
    viki_indices = df[viki_mask].index.tolist()
    viki_embeddings = embeddings[viki_indices]
    
    print(f"    Vikidia (simple prototype): {len(viki_indices):,}")
    
    simple_centroid = viki_embeddings.mean(axis=0, keepdims=True)
    
    similarities = cosine_similarity(embeddings, simple_centroid).flatten()
    
    print(f"    Similarity range: [{similarities.min():.4f}, {similarities.max():.4f}]")
    print(f"    Similarity mean: {similarities.mean():.4f}")
    
    return similarities.astype(np.float32)


# Main Feature Extraction with Checkpoints
def extract_all_features_with_checkpoints(
    df: pd.DataFrame,
    lang: str,
    hw_config: Dict,
    checkpoint_dir: Path
) -> pd.DataFrame:
    """Extract all features with checkpoint support."""
    print(f"\nStep 6-7: Extracting features for ENTIRE dataset")
    
    sentences = df['Sentence'].tolist()
    n_sentences = len(sentences)
    print(f"  Total sentences: {n_sentences:,}")
    
    # Determine number of CPU cores to use (for basic features only)
    n_jobs = min(hw_config['n_cpus'], 4)
    print(f"  CPU cores for basic features: {n_jobs}")
    print(f"  spaCy mode: single-thread (Notebook compatibility)")
    
    # Basic Features (CPU Parallel via joblib)
    if checkpoint_exists(checkpoint_dir, lang, 'basic'):
        print("\n  [CHECKPOINT] Loading basic features...")
        basic_df = load_checkpoint_csv(checkpoint_dir, lang, 'basic')
    else:
        print("\n  Extracting basic features (CPU parallel)...")
        basic_df = extract_basic_features(sentences, n_jobs=n_jobs)
        save_checkpoint_csv(basic_df, checkpoint_dir, lang, 'basic')
    
    # spaCy Features (Single-thread for Notebook compatibility)
    if checkpoint_exists(checkpoint_dir, lang, 'spacy'):
        print("\n  [CHECKPOINT] Loading spaCy features...")
        spacy_df = load_checkpoint_csv(checkpoint_dir, lang, 'spacy')
    else:
        print("\n  Extracting spaCy features (single-thread)...")
        nlp = load_spacy_model(lang)
        spacy_df = extract_spacy_features(sentences, nlp)
        del nlp
        gc.collect()
        save_checkpoint_csv(spacy_df, checkpoint_dir, lang, 'spacy')
    
    # Sentence-BERT Embeddings (Single GPU)
    if checkpoint_exists(checkpoint_dir, lang, 'sbert', 'npy'):
        print("\n  [CHECKPOINT] Loading SBERT embeddings...")
        embeddings = load_checkpoint_npy(checkpoint_dir, lang, 'sbert')
    else:
        print("\n  Computing SBERT embeddings (GPU accelerated)...")
        embeddings = compute_sbert_embeddings(sentences, lang, hw_config)
        save_checkpoint_npy(embeddings, checkpoint_dir, lang, 'sbert')
    
    # Cosine Similarity
    cos_simi = compute_cosine_similarity_from_embeddings(embeddings, df)
    
    del embeddings
    gc.collect()
    if hw_config['device'] == 'cuda':
        torch.cuda.empty_cache()
    
    # Additional Features
    print("\n  Computing additional features...")
    words_chars_ratio = (df['LengthWords'] / df['LengthChars'].replace(0, 1)).astype(np.float32)
    
    # Build Final DataFrame
    print("\n  Building final feature DataFrame...")
    
    result_df = pd.DataFrame({
        'Index': df['Index'].values,
        'ID': df['ID'].values,
        'Name': df['Name'].values,
        'Sentence': df['Sentence'].values,
        'Label': df['Label'].values,
        'LengthWords': df['LengthWords'].values,
        'LengthChars': df['LengthChars'].values,
        'words_chars_ratio': words_chars_ratio.values,
        'cos_simi': cos_simi
    })
    
    for col in BASIC_FEATURE_NAMES:
        result_df[col] = basic_df[col].values
    
    for col in SPACY_FEATURE_NAMES:
        result_df[col] = spacy_df[col].values
    
    print(f"  Final shape: {result_df.shape}")
    
    del basic_df, spacy_df
    gc.collect()
    
    return result_df


# Step 8: Save Final Features
def save_features(df: pd.DataFrame, lang: str, output_dir: Path) -> Path:
    print(f"\nStep 8: Saving final features")
    
    output_file = output_dir / f"{lang}_full_features.csv"
    
    df.to_csv(output_file, index=False)
    print(f"  Saved: {output_file}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    
    return output_file


# Print Reports
def print_cleaning_report(lang: str, stats: Dict) -> None:
    print("\n")
    print("=" * 70)
    print(f"CLEANING REPORT - {lang.upper()}")
    print("=" * 70)
    
    print(f"\nOriginal rows: {stats['original_rows']:,}")
    
    print(f"\nBasic cleaning:")
    print(f"  NaN removed: {stats['nan_removed']:,}")
    print(f"  Blank removed: {stats['blank_removed']:,}")
    
    print(f"\nDuplicates removed:")
    print(f"  Vikidia internal: {stats['viki_internal_dup']:,}")
    print(f"  Wikipedia internal: {stats['wiki_internal_dup']:,}")
    print(f"  Leakage (Viki ∩ Wiki): {stats['leakage_removed']:,}")
    
    print(f"\nAfter cleaning:")
    print(f"  Vikidia: {stats['viki_final']:,}")
    print(f"  Wikipedia: {stats['wiki_final']:,}")
    print(f"  Total: {stats['after_dedup']:,}")
    
    print("=" * 70)


def print_feature_summary(df: pd.DataFrame, lang: str) -> None:
    print("\n")
    print("=" * 70)
    print(f"FEATURE SUMMARY - {lang.upper()}")
    print("=" * 70)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    numeric_cols = ['LengthWords', 'LengthChars', 'words_chars_ratio', 'cos_simi'] + \
                   BASIC_FEATURE_NAMES + SPACY_FEATURE_NAMES
    
    print(f"\nFeature statistics:")
    print(df[numeric_cols].describe().round(3).to_string())
    
    print("=" * 70)


# Main Pipeline
def process_language(
    lang: str, 
    data_dir: Path, 
    output_dir: Path, 
    checkpoint_dir: Path,
    hw_config: Dict
) -> None:
    print("\n" + "=" * 70)
    print(f"PROCESSING {lang.upper()} DATASET")
    print("=" * 70)
    
    start_time = time.time()
    
    # Check if we have cleaned data checkpoint
    if checkpoint_exists(checkpoint_dir, lang, 'cleaned'):
        print("\n[CHECKPOINT] Loading cleaned dataset...")
        df = load_checkpoint_csv(checkpoint_dir, lang, 'cleaned')
        # Reconstruct stats (approximate)
        stats = {
            'original_rows': len(df),
            'nan_removed': 0,
            'blank_removed': 0,
            'viki_internal_dup': 0,
            'wiki_internal_dup': 0,
            'leakage_removed': 0,
            'after_dedup': len(df),
            'viki_final': len(df[df['source'] == 'viki']),
            'wiki_final': len(df[df['source'] == 'wiki'])
        }
        print(f"  Loaded {len(df):,} rows from checkpoint")
    else:
        # Step 1: Load data
        df = load_data(lang, data_dir)
        
        # Step 2: Assign unique Index
        df = assign_unique_index(df)
        
        # Step 3: Basic cleaning
        df, stats = basic_cleaning(df)
        
        # Step 4: Remove duplicates
        df, stats = remove_duplicates(df, stats)
        
        # Save cleaned checkpoint
        save_checkpoint_csv(df, checkpoint_dir, lang, 'cleaned')
        
        # Step 5: Save cleaned dataset
        save_cleaned_dataset(df, lang, output_dir)
    
    # Step 6-7: Extract all features
    result_df = extract_all_features_with_checkpoints(df, lang, hw_config, checkpoint_dir)
    
    # Step 8: Save final features
    save_features(result_df, lang, output_dir)
    
    # Print reports
    print_cleaning_report(lang, stats)
    print_feature_summary(result_df, lang)
    
    total_time = time.time() - start_time
    print(f"\nTotal processing time: {total_time/60:.1f} minutes")
    
    # Clear checkpoints after successful completion
    print("\nClearing checkpoints...")
    clear_checkpoints(checkpoint_dir, lang)
    
    # Cleanup
    del df, result_df
    gc.collect()
    if hw_config['device'] == 'cuda':
        torch.cuda.empty_cache()


def main():
    """Main entry point."""
    print("\n" + "=" * 70)
    print("iDEM FEATURE EXTRACTION PIPELINE")
    print("Kaggle Notebook Version (T4 GPU + Single-thread spaCy)")
    print("** WITH CHECKPOINT SUPPORT **")
    print("=" * 70)
    
    # Check environment and get hardware config
    hw_config, is_kaggle = check_environment()
    
    # Print detailed hardware info
    print_hardware_info(hw_config)
    
    # Setup paths
    data_dir, output_dir, checkpoint_dir = setup_paths(is_kaggle)
    
    # Check for existing checkpoints
    existing_checkpoints = list(checkpoint_dir.glob("checkpoint_*"))
    if existing_checkpoints:
        print(f"\nFound {len(existing_checkpoints)} existing checkpoints:")
        for cp in existing_checkpoints:
            print(f"  {cp.name}")
        print("Will resume from checkpoints where possible.\n")
    
    # Process both languages
    for lang in ['en', 'fr']:
        try:
            process_language(lang, data_dir, output_dir, checkpoint_dir, hw_config)
        except FileNotFoundError as e:
            print(f"\nSkipping {lang}: {e}")
        
        gc.collect()
        if hw_config['device'] == 'cuda':
            torch.cuda.empty_cache()
    
    print("\n" + "=" * 70)
    print("ALL PROCESSING COMPLETE!")
    print("=" * 70)
    
    print("\nOutput files:")
    for f in sorted(output_dir.glob("*.csv")):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name} ({size_mb:.1f} MB)")


# Kaggle Notebook Cell Execution
if __name__ == "__main__":
    main()