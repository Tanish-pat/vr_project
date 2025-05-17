# Baseline Evaluation for Zero-Shot VQA

This folder contains the baseline evaluation results for a zero-shot Visual Question Answering (VQA) model.

## Contents

### 1. `vqa_evaluation_results.csv.zip`

This compressed CSV file contains the **per-question evaluation results** for the zero-shot VQA setup.  
Each row corresponds to a question-answer (QA) pair with associated metric outputs (e.g., BERTScore, Exact Match, etc.).

### 2. `Results/`

This folder includes:

- `vqa_metrics.npz`: A compressed NumPy archive containing metric arrays (Exact Match, BERT Score components, Sentence Similarity, etc.) for all QA pairs.
- `main.py` : A Python script that loads the `.npz` file and calculates **average performance metrics** across the entire evaluation set.



