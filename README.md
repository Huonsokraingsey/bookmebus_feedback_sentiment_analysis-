# Sentiment Analysis of Khmer-Language Customer Reviews
### A Tool for Local Business Feedback Mining

**ITM 454 – Natural Language Processing** · American University of Phnom Penh, Faculty of Digital Technologies · Fall 2026

**Team:** HUON Sokraingsey, KHLIM Vinchhieng, HANG Lifong, UNG Seangeang

**Instructor:** Ms. Monyrath Buntoun

**GitHub repository:** https://github.com/Huonsokraingsey/bookmebus_feedback_sentiment_analysis-

---

## Overview

Khmer is a low-resource language for NLP. It is written without spaces between words, has very few annotated corpora, and has almost no public sentiment tools. Cambodian businesses receive more and more customer feedback in Khmer, but English-language sentiment tools cannot read it.

This project builds a **Khmer sentiment classifier** for customer reviews (positive / negative; see *Changes from the proposal* for why neutral was dropped) and compares three model families on the same data:

| Family | Model |
|---|---|
| Statistical ML | TF-IDF + Logistic Regression |
| Neural network | BiLSTM with pretrained fastText Khmer embeddings |
| Transformer | Fine-tuned XLM-R (`xlm-roberta-base`) |

As an add-on, a **Streamlit web app** lets a business upload its feedback as CSV or Excel and see a sentiment dashboard, including which words are most often linked to positive or negative feedback.

---

## Dataset

| | |
|---|---|
| Source | [seanghay/bookmebus-reviews](https://huggingface.co/datasets/seanghay/bookmebus-reviews) on Hugging Face |
| Content | Customer reviews of BookMeBus, a Cambodian bus, van and ferry booking platform |
| Working file | `data/bookmebus_reviews_khmer.csv` – 4,114 rows |
| Columns | `text` (original review), `score` (0–5, steps of 0.2), `Translation` (Khmer translation of `text`) |
| Translations | The Hugging Face dataset only has `text` and `score`. We added the `Translation` column with the **OpenAI API (via Codex)** |

Notes from our inspection:
- Most reviews are in English; **230** contain Khmer characters and are the candidates for our native-Khmer test set.
- **3** rows have no review text (their `Translation` is a placeholder word, not a real translation).
- **213** rows have a `Translation` identical to `text` (Khmer originals, emoji-only or number-only reviews).
- **2** rows have `text = "#NAME?"`: Excel turned reviews starting with `+` / `-` into a formula error, so the original text is lost (excluded).
- **10** reviews are written in Japanese, Chinese, Russian or Korean (labeled from the original; the model uses their Khmer translation).
- Scores are skewed: **1,014** reviews have a score of 5.0. Scores often disagree with the written sentiment, which is why we created new labels (see below).

The original CSV is never modified.

**License note:** the Hugging Face dataset card does not state a license. The reviews (and the files derived from them in `data/` and `results/predictions/`) are included here only for academic, non-commercial use in this course project, with attribution to the dataset author. This repository itself has no license.

---

## Method

### 1. Labels
Score-derived labels were abandoned because ratings often disagree with what reviews say (for example, a polite complaint with a high score).

- **Silver labels:** every review was labeled from its text by Claude (Anthropic) following written guidelines (`data/labels/LABELING_GUIDELINES.md`), including rules for indirect complaints, negation, mixed reviews, sarcasm and emoji-only reviews. After a 50-review pilot and a full pass, all 1,919 borderline reviews were re-labeled **blind** with extra consistency rules (R1–R9); the first pass and the review agreed on 91% of them (`data/labels/label_report.md`).
- Result: 1,966 negative, 1,898 positive, 234 neutral, 16 excluded. Only 5.7% of the reviews (9 of the Khmer reviews) are neutral, so the models use **2 classes** (positive / negative).
- **Gold labels:** the native-Khmer reviews used for testing were labeled independently by two team members (`data/labels/ANNOTATION_INSTRUCTIONS.md`). They agreed on **96.4%** of the 222 reviews both marked usable (**Cohen's kappa 0.902**); the 8 disagreements were decided together (`data/labels/gold_disagreements.csv` → `gold_test_labels.csv`). The silver labels agree with the gold labels on **98.2%** of these reviews.
- **Human check of the silver labels:** on 150 random reviews (50 per AI label), a team member who did not see the AI labels agreed with them on **98.0%** of the reviews both called positive or negative, but confirmed only 54% of the AI's *neutral* labels (`data/labels/human_review_results.md`).
- Notebook 01 compares the old score-based labels with the new labels: they agree on only **62.3%** of the reviews.

### 2. Preprocessing (notebook 01)
Unicode normalization, removal of zero-width characters and URLs, collapsing repeated characters, lowercasing English words in code-switched text, removing punctuation and numbers, and Khmer word segmentation with **khmer-nltk**. Emojis and negation words (មិន, ទេ, គ្មាន, កុំ, ពុំ, not, no) are **kept**, because they carry sentiment; no stopwords are removed. The function is saved in `src/preprocess.py` so the web app uses exactly the same code. XLM-R uses its own subword tokenizer on the normalized text instead.

### 3. Data splits

| Set | Content | Size | Used for |
|---|---|---|---|
| Train (80%) | Khmer translations of English reviews | 2,858 | learning the weights |
| Validation (10%) | Khmer translations | 357 | choosing hyperparameters and the best epoch |
| Test – translated (10%) | Khmer translations | 358 | final score on translated text |
| Test – native | Original Khmer reviews with **gold** labels (193 pure Khmer, 21 code-switched) | 214 | final score on real Khmer text |

Duplicates are removed before splitting, native-Khmer reviews are never used for training or tuning, and each test set is used only once per model. The translated sets are balanced (50% positive); test_native is only 20.6% positive.

### 4. Evaluation
- **Primary metric:** macro-F1 (the classes are imbalanced).
- Also: accuracy, per-class precision / recall / F1, confusion matrices, and bootstrap 95% confidence intervals for the native test set.
- Breakdowns: translated vs. native test data, and pure-Khmer vs. code-switched reviews.
- Qualitative error analysis of misclassified reviews (negation, indirect complaints, mixed sentiment, sarcasm, code-switching, translation problems).

---

## Project structure

```
nlp_project/
├── data/
│   ├── bookmebus_reviews_khmer.csv        # input (never modified)
│   ├── labels/                            # silver labels, guidelines, label report, gold annotation files
│   │   └── work/                          # labeling work files (shards, review set) used by scripts/
│   ├── processed/                         # train / val / test_translated / test_native + settings.json (notebook 01)
│   └── demo/sample_feedback.csv           # test-set reviews for the web app demo (Part C)
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_statistical_ml_tfidf_logreg.ipynb
│   ├── 03_nn_bilstm.ipynb
│   ├── 04_transformer_xlmr.ipynb
│   └── 05_comparison_and_error_analysis.ipynb
├── scripts/
│   ├── label_batch.py, review_batch.py, finalize_labels.py   # Part A: silver labels
│   ├── human_vs_ai_labels.py              # human check of the silver labels → data/labels/human_review_results.md
│   ├── make_demo_data.py                  # data/demo/sample_feedback.csv
│   ├── make_keyword_review.py             # keyword review sheet for the web app's Khmer word lists
│   ├── ablation_tfidf.py                  # pre-processing and class-weight ablations (validation set only)
│   └── make_slide_figures.py              # pipeline and error-category figures for the presentation
├── src/preprocess.py                      # same preprocessing as notebook 01 (written by it), used by the app
├── app/                                   # Streamlit web app (add-on, Part C)
├── models/                                # only tfidf_logreg.joblib is in git (see "Large files")
├── human-review/                          # the team's original annotation and review files, unchanged
├── results/
│   ├── predictions/                       # one prediction per test review and model
│   ├── metrics/                           # scores per model + ablation.json
│   └── figures/
├── tests/                                 # pytest tests for preprocessing and the web app helpers
├── requirements.txt
└── README.md
```

---

## Setup

Requires **Python 3.12 or newer**. The results were produced with **Python 3.13.15** on Windows 11, PyTorch 2.14.1 + CUDA 12.6 on an NVIDIA RTX 4060 laptop GPU (8 GB). `requirements.txt` pins the exact package versions we used.

```powershell
git clone https://github.com/Huonsokraingsey/bookmebus_feedback_sentiment_analysis-.git nlp_project
cd nlp_project
python -m venv .venv
.venv\Scripts\Activate.ps1                       # Mac/Linux: source .venv/bin/activate
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126   # GPU version of PyTorch (skip without an NVIDIA GPU)
pip install -r requirements.txt
```

Every notebook starts with a **setup cell** that goes to the project folder and runs `%pip install -r requirements.txt` (nothing is installed if the packages are already there), followed by a table of the input files it needs and the notebook that creates them.

**Additional notes:**
- **GPU:** notebook 04 (XLM-R) needs an NVIDIA GPU (about 2 minutes on the RTX 4060). Notebooks 01, 02 and 05 run in seconds on a CPU; notebook 03 (BiLSTM) is faster with a GPU but also runs on a CPU.
- **Windows:** keep the project in a short folder path (e.g. `Desktop\nlp_project`). JupyterLab contains very long file names, and pip fails with "No such file or directory … Long Path support" when the full path exceeds 260 characters.
- **Khmer font for plots:** install Noto Sans Khmer (or Khmer OS) so Khmer text in matplotlib figures does not appear as empty boxes.

### Large files

These files are **not in the repository** (GitHub rejects files over 100 MB). The notebooks recreate them:

| File | Size | How to get it |
|---|---|---|
| `models/cc.km.300.vec.gz` (fastText Khmer word vectors) | 145 MB | downloaded automatically by notebook 03 on its first run from [fastText: Word vectors for 157 languages](https://fasttext.cc/docs/en/crawl-vectors.html) (Grave et al., 2018) |
| `models/xlmr_best/` (fine-tuned XLM-R) | 1.1 GB | created by notebook 04 (downloads `xlm-roberta-base` from Hugging Face, then fine-tunes it). Needed only for the web app's **Accurate** model |

`models/tfidf_logreg.joblib` (2.4 MB, created by notebook 02) **is** in the repository, so the web app works right after installing.

All saved notebook outputs, metrics (`results/metrics/`), predictions and figures are in the repository, so the results can be checked without retraining. Retraining the BiLSTM or XLM-R on a different GPU can give slightly different numbers.

---

## How to run

Run the steps in this order:

1. **Labels (already done, in the repository):** `data/labels/bookmebus_labeled.csv` is the Part A output (`scripts/finalize_labels.py` rebuilds it from `labels_progress.csv`, `labels_review.csv` and `work/review_set.csv`). The gold labels of the native test set are in `gold_test_annotation_A.csv`, `gold_test_annotation_B.csv` and `gold_disagreements.csv` (see `data/labels/ANNOTATION_INSTRUCTIONS.md`).
2. `01_data_preparation.ipynb` – cleaning, label comparison, gold-label agreement, splits.
3. `02_statistical_ml_tfidf_logreg.ipynb`
4. `03_nn_bilstm.ipynb`
5. `04_transformer_xlmr.ipynb` (GPU)
6. `05_comparison_and_error_analysis.ipynb` – comparison table, charts, error analysis.

Each model notebook saves its predictions to `results/predictions/`, its scores to `results/metrics/` and its figures to `results/figures/`. All notebooks use seed 45 and should run top to bottom with "Restart & Run All".

Run the tests with:

```bash
pytest tests/
```

---

## Results

The final comparison of the three models is produced by `05_comparison_and_error_analysis.ipynb` from the files in `results/metrics/`. The notebook states whether the native test set used **gold** (human) or **silver** (AI) labels; results on silver labels are provisional.

The native test set uses **gold** (human) labels: two annotators, Cohen's kappa 0.902.

| | TF-IDF + LogReg | BiLSTM + fastText | **XLM-R (fine-tuned)** |
|---|---|---|---|
| Model type | Statistical ML | Neural network | Transformer |
| Trainable weights | 65,481 | 908,998 | 278,045,186 |
| Best validation macro-F1 | 0.938 | 0.910 | **0.950** |
| test_translated macro-F1 (accuracy) | 0.964 (0.964) | 0.919 (0.919) | **0.975 (0.975)** |
| test_native macro-F1 (accuracy), gold labels | 0.851 (0.893) | 0.680 (0.715) | **0.950 (0.967)** |
| test_native macro-F1, 95% bootstrap CI | [0.792, 0.903] | [0.608, 0.741] | [0.909, 0.981] |
| pure Khmer / code-switched macro-F1 | 0.861 / 0.659 | 0.708 / 0.430 | 0.962 / 0.724 |

- **Best model: fine-tuned XLM-R.** On real Khmer reviews its confidence interval does not overlap with the other models'.
- **Translation gap** (test_native − test_translated): TF-IDF −0.113, BiLSTM −0.239, XLM-R −0.025. Pretraining on real Khmer text and subword pieces almost close the gap (the BiLSTM does not know 20.8% of the words in real Khmer reviews).
- **The BiLSTM is worse than the TF-IDF baseline**: a small training set and a word vocabulary built from translations.
- **Remaining XLM-R errors** (16 in total, 7 on real Khmer reviews): mostly mixed reviews and polite/indirect complaints (e.g. "you should choose a hygienic place to eat"); 2 of them (in test_translated) may be errors of the silver labels.

- **Ablations** (validation set only, `scripts/ablation_tfidf.py` → `results/metrics/ablation.json`): Khmer word segmentation matters most (word TF-IDF macro-F1 0.924 → 0.776 without it); removing negation words or stopwords lowers the score; class weights make no difference because the training set is balanced.

Figures are in `results/figures/`, predictions per review in `results/predictions/`.

---

## Web app (add-on)

The web app demonstrates a practical use of the classifier for local businesses. It only uses the models trained in the notebooks; it does not retrain anything or produce results for the report.

```bash
streamlit run app/app.py
```

**It needs:** `models/tfidf_logreg.joblib` (from notebook 02) and, for the "Accurate" option, `models/xlmr_best/` (from notebook 04). The sample data `data/demo/sample_feedback.csv` contains only test-set reviews without labels (`python scripts/make_demo_data.py` recreates it). The app code is in `app/`: `app.py` (page layout), `sentiment.py` (loads the models, predicts in batches, explains TF-IDF predictions), `indicators.py` (word statistics), `aspects.py` (keyword lists of the aspect summary), `file_io.py` (reading CSV/Excel), `stopwords_km.txt` (Khmer function words hidden from the word charts only).

The app has two tabs:
- **📄 Analyse a file** – the dashboard: overview cards, sentiment distribution (+ per category), trend over time (if the file has a date column), most mentioned words, strongest indicator words, word explorer, aspect summary (keyword matching, *not* a trained aspect model), review table and download.
- **✍️ Analyse one review** – type or paste one review (or pick one of the 5 example sentences from the notebooks) and see the prediction of **both** models side by side; each word is colored by how much it pushed the TF-IDF model towards positive (green) or negative (red).

The models are 2-class, so every review is **positive** or **negative**; when the model is less than 60% sure, the review is shown as **uncertain**. For the same reviews, the app gives exactly the same predicted labels as the notebooks.

**User guide for a business owner:**
1. Upload a CSV or Excel file of customer feedback (up to 10 MB / 5,000 rows), or click **Try sample data**.
2. Choose the column that contains the feedback text (the app guesses it). Optionally choose a date column (trend chart) and a category column (for example, branch or route) to compare and filter.
3. Choose a model: **Fast** (TF-IDF, explains each prediction with its top words) or **Accurate** (XLM-R, slower on a normal computer). Each model shows its test score next to its name.
4. Read the dashboard: share of positive / negative / uncertain feedback, the most mentioned words and whether they are mostly good or bad, the words most linked to negative or positive feedback, and every review with its prediction (filter by sentiment, confidence or keyword, e.g. កៅអី "seat").
5. Download the results as CSV (opens in Excel with Khmer letters) or Excel, with the added columns `sentiment`, `confidence`, `prob_negative`, `prob_positive` and `language`.
6. To check a single review, open the **✍️ Analyse one review** tab and type it.

Uploaded files stay in memory for your session only and are not saved. The model was trained on Khmer bus and ferry reviews, so it is most reliable for Khmer text in similar domains; English text and indirect or sarcastic complaints are less reliable.

---

## Changes from the proposal

| Proposal | Final project | Reason |
|---|---|---|
| Labels from scores (weak supervision), checked against a gold subset | Silver labels from the review text (Claude), gold human labels for the native test set (two annotators, kappa 0.902) | Scores often disagree with the written sentiment (only 62.3% agreement) |
| 3-class polarity (positive / neutral / negative) | 2 classes (positive / negative); neutral reviews stay labeled in `bookmebus_labeled.csv` but are not used by the models | Only 5.7% of the reviews were neutral, and only 9 of the Khmer test reviews: too few to learn or to measure reliably |
| Gemini-generated Khmer translations | Khmer translations made with the OpenAI API (via Codex), stored in the `Translation` column; problems are flagged, not retranslated | We used the OpenAI API instead of Gemini for the translation step |
| TF-IDF baseline vs. XLM-R / LaBSE | TF-IDF + Logistic Regression, BiLSTM, and XLM-R | Compare all three model families; LaBSE not used |
| Compare native vs. translated **training** data | Not run | All ~230 native-Khmer reviews are needed for the test set |
| Aspect-level sentiment (stretch goal) | Keyword-based aspect summary in the web app only | Not a trained aspect model; not part of the research results |

---

## Limitations

- The native-Khmer test set is small (214 reviews, only 44 positive and 21 code-switched), so differences between models may not be statistically reliable; see the confidence intervals in notebook 05.
- Training data is machine-translated, so it may not reflect how Cambodians naturally write reviews.
- The training, validation and translated-test labels are AI-generated silver labels and may contain errors (humans confirmed 98% of the positive/negative ones). Our strict safety rule (a named safety problem makes a review negative) caused some debatable labels. Only the native test set has gold labels.
- Neutral reviews were dropped (2 classes), and each model was trained once with one seed (45).
- All reviews come from one platform and domain (bus, van and ferry travel).
- Indirect complaints, sarcasm and code-switched text remain difficult.

---

## AI-use statement

We used generative AI tools in this project:

| Tool | Used for |
|---|---|
| **Claude** (Anthropic; Claude Opus 5.5 in Claude Code) | the silver sentiment labels (Part A, following `data/labels/LABELING_GUIDELINES.md`); drafting the project code, notebooks, web app, tests and documentation from our project instructions |
| **OpenAI API** (via Codex) | the Khmer translations in the `Translation` column of `data/bookmebus_reviews_khmer.csv` |

All AI-generated code and text was reviewed by the team. The gold labels for the native-Khmer test set, the human check of the silver labels and the review of the web app's Khmer word lists were done by team members.

- Anthropic. (2026). *Claude* (Opus 5.5) [Large language model]. https://www.anthropic.com/claude
- OpenAI. (2026). *OpenAI API* [Large language model API, used via Codex]. https://platform.openai.com

---

## Acknowledgments and references

- Seanghay. *bookmebus-reviews* [Data set]. Hugging Face. https://huggingface.co/datasets/seanghay/bookmebus-reviews
- khmer-nltk: Khmer natural language processing toolkit.
- Conneau, A., et al. (2020). Unsupervised cross-lingual representation learning at scale. *ACL 2020*.
- Grave, E., et al. (2018). Learning word vectors for 157 languages. *LREC 2018*.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735–1780.
- Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement, 20*(1), 37–46.
- Notebook structure follows the ITM 454 course notebooks.
