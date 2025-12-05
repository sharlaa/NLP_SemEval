import numpy as np
import math
from scipy.stats import pearsonr
import torch

def evaluate_predictions(pred_a, pred_v, gold_a, gold_v):
    squared_errors = (pred_v - gold_v)**2 + (pred_a - gold_a)**2
    rmse_va = math.sqrt(np.mean(squared_errors))

    return {
        "RMSE_VA": rmse_va,
        "PCC_V": pearsonr(pred_v, gold_v)[0],
        "PCC_A": pearsonr(pred_a, gold_a)[0],
        "RMSE_V": math.sqrt(np.mean((pred_v - gold_v) ** 2)),
        "RMSE_A": math.sqrt(np.mean((pred_a - gold_a) ** 2)),
    }


def get_predictions(model, dataloader, device):
    preds, labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device)
            ).cpu().numpy()
            golds = batch["labels"].cpu().numpy()
            preds.append(outputs)
            labels.append(golds)
    preds, labels = np.vstack(preds), np.vstack(labels)
    return preds[:,0], preds[:,1], labels[:,0], labels[:,1]
