# How to fill in the gold annotation file

Thank you for helping! You will label **228 short customer reviews written in Khmer** (bus / van / ferry trips
booked on BookMeBus). Your labels become the **gold (human) labels** of our final test set.

## Which file?

| Person | File |
|---|---|
| Annotator **A** | `data/labels/gold_test_annotation_A.csv` |
| Annotator **B** | `data/labels/gold_test_annotation_B.csv` |

Work **alone**: do not look at the other person's file, do not discuss the reviews before you are both finished,
and do not look at `bookmebus_labeled.csv` (it contains the AI labels).

## How to open it

- **Excel**: double-click the file. Khmer should display correctly.
- **Google Sheets**: File → Import → Upload → choose the file → "Replace spreadsheet".

## What to fill in (one row per review)

| Column | What to write |
|---|---|
| `row_id` | **do not change** |
| `text` | the review – read it, **do not change** it |
| `label` | `positive`, `negative` or `neutral` (lowercase, English) |
| `is_usable` | `yes` = it is a real review you can judge · `no` = not a review / meaningless / you cannot understand it (then `label` may stay empty) |
| `notes` | optional, e.g. "sarcasm?", "mixed", "not sure" |

## How to decide the label

Ask yourself: **"What would the bus company owner take away from this review?"**

- **positive** – the customer is satisfied or praises the service.
- **negative** – the customer complains or reports a problem, **even if it is polite or indirect**
  (e.g. "thank you, even though the bus was 2 hours late" → negative; "the driver should be more patient" → negative).
  A safety problem (dangerous driving, driver on the phone) → negative.
- **neutral** – no opinion at all (a question, booking information), or praise and complaint of about equal weight.

Read only the review text. If a positive review contains a small suggestion, it stays positive.
Full guidelines: `data/labels/LABELING_GUIDELINES.md` (sections 1 and 3).

It takes about **1 hour** (most reviews are one or two sentences).

## How to save it (important for Khmer letters!)

- **Excel**: File → Save As → type **"CSV UTF-8 (Comma delimited) (*.csv)"** → keep the same file name.
  ⚠ Do **not** choose plain "CSV (Comma delimited)": it destroys the Khmer letters.
- **Google Sheets**: File → Download → **Comma-separated values (.csv)**, then rename it to the original file name.

Send the file back (or put it in `data/labels/` with the same name, replacing the empty one).

## What happens next (for the team)

When **both** files are filled in, run `notebooks/01_data_preparation.ipynb` again. It measures how well A and B agree
(Cohen's kappa) and saves the reviews where you disagree to `data/labels/gold_disagreements.csv`. Decide those
together in the `final_label` column, run notebook 01 once more, then run notebooks 02–05 again: the native-Khmer
test results will then use your gold labels instead of the AI labels.
