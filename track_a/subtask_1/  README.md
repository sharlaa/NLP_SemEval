# Track A – Subtask 1 (English, Valence–Arousal Regression)

This folder contains the full pipeline for Subtask 1 on the English
laptop and restaurant domains.

## Data

Place the four official JSONL files in:

- `data/eng_laptop_train_alltasks.jsonl`
- `data/eng_laptop_dev_task1.jsonl`
- `data/eng_restaurant_train_alltasks.jsonl`
- `data/eng_restaurant_dev_task1.jsonl`

(These are already in the repo for internal use and are not modified.)

## Code Structure

- `src/data_loader.py` – JSONL → DataFrame conversion and `VADataset`
- `src/model.py` – `TransformerVARegressor` (Transformer backbone + regression head)
- `src/train.py` – training loop, early stopping, evaluation
- `src/utils.py` – helper functions for RMSE_VA, PCC, etc.
- `src/inference.py` – generates JSONL submission files

## Notebook

`notebooks/modeling_subtask1_eng.ipynb` runs the entire pipeline for:

- Domain **laptop**
- Domain **restaurant**

For each domain it:

1. Loads the local JSONL data
2. Splits the train set into train/dev (90/10)
3. Trains multiple Transformer models:
   - `bert-base-uncased`
   - `j-hartmann/emotion-english-distilroberta-base`
4. Selects the best model according to `RMSE_VA`
5. Generates predictions for the dev set:
   - `submissions/eng_laptop_dev_task1_predictions.jsonl`
   - `submissions/eng_restaurant_dev_task1_predictions.jsonl`

These prediction files are in the official submission format for Subtask 1.
