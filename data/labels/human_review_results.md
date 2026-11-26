# Human check of the AI (silver) labels

150 random reviews (50 per AI label, shuffled), labeled by team members **without seeing the AI label**. Annotators who filled in every row: annotator_A.

## annotator_A vs. AI

- Agreement (3 classes): **82.7%**, Cohen's kappa **0.740**
- Agreement when both say positive or negative (the 2 classes the models use): **98.0%** (99 reviews)
- Share of each AI label that the human confirmed: negative 100%, neutral 54%, positive 94%

| AI label   |   negative |   neutral |   positive |
|:-----------|-----------:|----------:|-----------:|
| negative   |         50 |         0 |          0 |
| neutral    |         14 |        27 |          9 |
| positive   |          2 |         1 |         47 |
