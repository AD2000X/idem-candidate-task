# iDem Research Task (EN/FR)

Welcome! This Task is part of the selection process for a **Research
Assistant** role on the iDem project (“Inclusive Democratic Spaces for
Deliberation and Participation”).

You are given **two CSV datasets**, one for English and one for French, built
from wikipedia and vikidea texts with sentence-level annotations for
complexity classification.

We are interested in:
- how you understand data  
- how you design and justify simple experiments  
- how you explain your methods and findings  

There is no single “correct” answer.

---
# Project Structure

```text
idem-candidate-task/
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
│  └─ 03_free_analysis.py
├─ notebooks/
│  └─ exploration.ipynb       # starter notebook (provided)
└─ reports/
   └─ presentation_outline.md # guidance for 10-slide presentation
```

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
- `Label` – binary label (0 = sentence annotated as **simple**,  
  1 = sentence annotated as **complex**).  
- `LengthWords` – sentence length in tokens
- `LengthChars` – sentence length in characters

More detail is in `data/README.md`.

---

## 2. Tasks

Please work through the tasks below. Focus on **clarity of reasoning** and
**reproducibility**.

### Task 0 – Quick data overview

Create a short overview of the datasets for each language:

1. Basic statistics:
   - number of sentences
   - distribution of `Label` (how many 0 / 1)
   - IQR range of sentence lengths.
2. Inspect a small sample of sentences and comment briefly:
   - How would *you* describe the difference between “complex” and
     “simple” sentences here?
   - Do you see obvious label noise or artefacts?

You can implement this in `src/01_data_overview.py` or a notebook.

---

## **Task 1 — Estimate the *true* proportion of simple sentences**

(including simple Vikidia-style sentences inside Wikipedia)

The `Label` column indicates whether a sentence is **simple** (`Label = 1`)
or **complex** (`Label = 0`). However, labels are **noisy**.

In particular:

* some “complex” sentences are actually simple
* some “simple” sentences are still complex
* **some Wikipedia sentences appear inside the Vikidia dataset**
  (we call this *leakage*)

Your goal is to estimate the **true proportion of simple sentences** in each dataset,
including the proportion of simple exact or Vikidia-style sentences that appear in Wikipedia.

1. **Compute the naïve estimate**:

   ```
   #(Label = 1) / total
   ```

2. **Design a better estimation method**, using any reasonable approach. 

   * similarity between "complex" Wikipedia sentences and simple Vikidia sentences
   * a simple classifier trained on clean subsets
   * your own method

3. **Produce final estimates**:

   * adjusted true proportion of simple sentences (in both EN and FR)
   * estimated proportion of Vikidia-like simple sentences inside Wikipedia
     (e.g. % of complex-labelled sentences that resemble simple sentences)

4. **Explain your reasoning**:

   * how you operationalised “simple”
   * how you detected Vikidia-style sentences
   * key assumptions & limitations
   * why your adjusted estimate is more realistic than the naïve one

Implement in:

```
src/02_estimate_simplified_proportion.py
```

---

### Task 2 – Do something interesting with the corpora (optional)

Choose **one** of the following (or propose your own small idea):

1. **Complex vs simple classifier**

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

Put code for this in `src/03_free_analysis.py` or a notebook.

---

# Deliverables

Please include:

### **1. Code & notebooks**

   - Place Python scripts in `src/` and any notebooks in `notebooks/`.
   - Code should be reasonably organised and runnable.
  
### **2. A presentation (PDF)**

  - Prepare a slide deck (up to **7 slides**), summarising:
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


Thank you for your time — we're looking forward to your submission.




