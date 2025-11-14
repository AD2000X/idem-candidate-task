# iDem Research Task (EN/FR)

Welcome! This mini-project is part of the selection process for a **Research
Assistant** role on the iDem project (“Inclusive Democratic Spaces for
Deliberation and Participation”).

You are given **two CSV datasets**, one for English and one for French, built
from wikipedia and vikidea texts with sentence-level annotations for
complexity classification.

We are interested in **how** you work:
- how you understand data  
- how you design and justify simple experiments  
- how clearly you explain your methods and findings  

There is no single “correct” answer.

---

## 1. Data

### Download data

The datasets used in this task can be downloaded from the GitHub Release:

https://github.com/Nouran-Khallaf/idem-candidate-task/releases/latest

Place both files into the `data/` folder before running the notebooks.

- `En-Dataset.csv` – English sentences  
- `Fr-Dataset.csv` – French sentences  

Each has the following columns:

- `ID` – document / article identifier
- `Name` – article title
- `Sentence` – a single sentence in that article (EN or FR)
- `Label` – binary label (1 = sentence annotated as **simplified**,  
  0 = sentence annotated as **complex**).  
- `LengthWords` – sentence length in tokens
- `LengthChars` – sentence length in characters

More detail is in `data/README.md`.

---

## 2. Tasks

Please work through the tasks below. Focus on **clarity of reasoning** and
**reproducibility**.

You may use either English, French, or both. When you choose only one language,
please say why.

### Task 0 – Quick data overview

Create a short overview of the datasets for each language:

1. Basic statistics:
   - number of sentences
   - distribution of `Label` (how many 0 / 1)
   - IQR range.
2. Inspect a small sample of sentences and comment briefly:
   - How would *you* describe the difference between “complex” and
     “simplified” sentences here?
   - Do you see obvious label noise or artefacts?

You can implement this in `src/01_data_overview.py` or a notebook.

---
Here is the **complete, polished, production-ready README.md** for your entire GitHub repo.

It incorporates:

✔ the full task description
✔ the Vikidea → Wikipedia “promotion” requirement
✔ instructions for data download (via Release)
✔ repo structure
✔ deliverables
✔ expectations
✔ clean formatting

You can **copy/paste** this directly into `README.md` on GitHub.

---

# 📘 iDem Research Assistant Task

## Lexical Simplification Analysis (English & French)

Welcome!
This repository contains the technical task for candidates applying to the **iDem Research Assistant position**, focusing on **lexical simplification**, **multilingual text processing**, and **methodological reasoning**.

Your goal is to explore noisy English/French corpora derived from **Wikipedia** and **Vikidia/Vikidea**, estimate simplification properties, build a small parallel corpus, and carry out a short focused analysis.

The purpose of this task is to understand **how you think**, not just whether you get “correct” numbers.

---

# 📁 Project Structure

```text
idem-ls-candidate-task/
├─ README.md                  # ← You are here
├─ requirements.txt
├─ data/
│  ├─ README.md               # instructions for downloading data
│  ├─ En-Dataset.csv          # (candidate downloads)
│  ├─ Fr-Dataset.csv
│  └─ processed/              # you create this directory for outputs
├─ src/
│  ├─ 01_data_overview.py
│  ├─ 02_estimate_simplified_proportion.py
│  ├─ 03_build_parallel_corpus.py
│  └─ 04_free_analysis.py
├─ notebooks/
│  └─ exploration.ipynb       # starter notebook (provided)
└─ reports/
   └─ presentation_outline.md # guidance for 10-slide presentation
```

---

# 📦 Data Download

Because the datasets are large, they are provided as downloadable release assets.

### 👉 Download both datasets from the latest GitHub Release:

**[https://github.com/USER/REPO/releases/latest](https://github.com/USER/REPO/releases/latest)**

You should download:

* `En-Dataset.csv`
* `Fr-Dataset.csv`

Place them inside the `data/` folder before running any scripts or notebooks.

More details about file format are in `data/README.md`.

---

# 🧠 Your Tasks

You may use either English, French, or both languages — if you choose only one, please briefly explain why.

---

## **Task 0 — Data Overview**

Start by familiarising yourself with the datasets.

Please:

1. Load the English and French CSV files.
2. Report basic statistics:

   * number of sentences
   * distribution of `Label`
   * average & median sentence length (`LengthWords`, `LengthChars`)
3. Inspect a small sample of sentences and comment briefly on:

   * differences between complex vs simplified sentences
   * potential label noise
   * any unusual artefacts or patterns

You may implement this in:

```
src/01_data_overview.py
```

or in a notebook.

---

## **Task 1 — Estimate the *true* proportion of simplified sentences**

(including simplified Vikidea-style sentences inside Wikipedia)

The `Label` column indicates whether a sentence is **simplified** (`Label = 1`)
or **complex** (`Label = 0`). However, labels are **noisy**.

In particular:

* some “complex” sentences are actually simple
* some “simplified” sentences are still complex
* **some simplified Vikidea/Vikidia sentences appear inside the Wikipedia dataset**
  (we call this *leakage*)

Your goal is to estimate the **true proportion of simplified sentences** in each dataset,
including the proportion of simplified exact or Vikidea-style sentences that appear in Wikipedia.

### Please:

1. **Compute the naïve estimate**:

   ```
   #(Label = 1) / total
   ```

2. **Design a better estimation method**, using any reasonable approach. 

   * similarity between "complex" Wikipedia sentences and simplified Vikidea sentences
   * a simple classifier trained on clean subsets
   * your own method

3. **Produce final estimates**:

   * adjusted true proportion of simplified sentences (EN/FR)
   * estimated proportion of Vikidea-like simplified sentences inside Wikipedia
     (e.g. % of complex-labelled sentences that resemble simplified sentences)

4. **Explain your reasoning**:

   * how you operationalised “simplified”
   * how you detected Vikidea-style sentences
   * key assumptions & limitations
   * why your adjusted estimate is more realistic than the naïve one

Implement in:

```
src/02_estimate_simplified_proportion.py
```

---

## **Task 2 — Build a parallel (complex → simplified) corpus**

Construct a small, cleaned set of `(complex_sentence, simplified_sentence)` pairs.

### Requirements:

1. **French is required.**
   English is optional but welcome.

2. Define an alignment method, for example:

   * aligning Vikidea simplified sentences with Wikipedia complex ones via similarity

3. Save your final corpus as:

```
data/processed/fr_parallel.tsv
data/processed/en_parallel.tsv   
```

with columns like:

| article_id | article_name | complex | simple | similarity_score |

4. Report:
   * how many pairs your method produced
   * what filters you used
   * your impression of quality (supported by a few examples)

Implement in:

```
src/03_build_parallel_corpus.py
```

---

### Task 3 – Do something interesting with the corpora

Choose **one** of the following (or propose your own small idea):

1. **Complex vs simplified classifier**

   - Train a simple model (e.g. logistic regression, small transformer, etc.)
     to predict the `Label` from `Sentence` (EN or FR).
   - Evaluate on a held-out test set.
   - Include **at least a little error analysis**:
     - show some misclassified sentences and comment on patterns.

2. **Complexity scoring**

   - Define a numeric “complexity” score per sentence.
   - Compare score distributions for Label 0 vs Label 1.
   - Discuss whether the scores match your intuition about complexity.

3. **Lexical simplification patterns**

   - Using your parallel corpus from Task 2, extract common **word-level substitutions** (complex → simple).
   - Provide some statistics and examples:
   - most frequent substitutions
   - types of words that tend to be simplified

Put code for this in `src/04_free_analysis.py` or a notebook.

---

# Deliverables

Please include:

### **1. Code & notebooks**

   - Place Python scripts in `src/` and any notebooks in `notebooks/`.
   - Code should be reasonably organised and runnable.
  
### **2. A presentation (PDF)**

  - Prepare a slide deck (up to **10 slides**), summarising:
     - problem understanding
     - methods
     - key results
     - main takeaways & limitations
   - Export slides as **PDF** and put in `reports/` (e.g. `reports/main_presentation.pdf`).

A slide outline template is provided in:

```
reports/presentation_outline.md
```
### 3. **Environment**

   - Include a `requirements.txt` (or `environment.yml`) listing your main
     dependencies.
     
---


# Evaluation Criteria

We are primarily evaluating:

* **Methodological clarity**
* **Ability to handle noisy data**
* **Justification of design choices**
* **Clean and reproducible code**
* **Interpretation of results**
* **Concise communication**

This is *not* about matching exact numbers — it's about showing how you think.

---


Thank you for your time — we look forward to your submission.




