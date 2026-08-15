# AI-text detection — evaluation results

_Protocol: leave-source-out 5-fold GroupKFold (classifier=GradientBoosting+sigmoid-calib, seed=0)_  
_Corpus: 20 paragraphs -> 44 sources -> 352 pseudo-samples (176 human / 176 ai)_

## Ablation (cross-validated out-of-fold)

| Model | AUC | AP | Acc | Precision | Recall | F1 | Brier |
|---|---|---|---|---|---|---|---|
| classical (fixed weights) | 0.663 | 0.667 | 0.517 | 0.714 | 0.057 | 0.105 | 0.274 |
| learned (handcrafted) | 0.760 ±0.061 | 0.811 | 0.724 | 0.776 | 0.631 | 0.696 | 0.183 |

## Per-artifact recall (learned model)

| Artifact | Recall | n |
|---|---|---|
| regularize_punctuation | 0.453 | 64 |
| flatten_burstiness | 0.457 | 46 |
| lexical_smoothing | 0.480 | 50 |
| inject_boilerplate | 0.965 | 57 |
| inject_repetition | 0.981 | 54 |

## External validation — real human vs ChatGPT (HC3)

_n = 300 human / 300 ai; learned model trained on the pseudo corpus only_

| Model | AUC | AP | Acc | F1 | Brier |
|---|---|---|---|---|---|
| learned | 0.825 | 0.751 | 0.725 | 0.781 | 0.204 |
| classical | 0.960 | 0.948 | 0.622 | 0.398 | 0.189 |
