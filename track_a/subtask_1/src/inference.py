import json
import numpy as np
import torch
from transformers import AutoTokenizer
from data_loader import VADataset
from model import TransformerVARegressor


def generate_submission(model, df, tokenizer, device, fname):

    grouped = df.groupby("ID")
    results = []

    for id_val, group in grouped:
        aspect_list = []

        for _, row in group.iterrows():
            text = f"{row['Aspect']}: {row['Text']}"

            enc = tokenizer(text, truncation=True, padding="max_length",
                            max_length=128, return_tensors="pt")

            with torch.no_grad():
                pred = model(
                    enc["input_ids"].to(device),
                    enc["attention_mask"].to(device)
                ).cpu().numpy()[0]

            v = round(float(np.clip(pred[0], 1.0, 9.0)), 2)
            a = round(float(np.clip(pred[1], 1.0, 9.0)), 2)

            aspect_list.append({
                "Aspect": row["Aspect"],
                "VA": f"{v:.2f}#{a:.2f}"
            })

        results.append({"ID": id_val, "Aspect_VA": aspect_list})

    with open(fname, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Saved: {fname}")
    return fname
