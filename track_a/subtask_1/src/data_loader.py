import json
import pandas as pd
from torch.utils.data import Dataset
import torch


def load_jsonl(path: str):
    """Load a local JSONL file into a list of Python dicts."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def jsonl_to_df(data):
    """
    Converts DimABSA JSONL formats into a unified DataFrame with:
    [ID, Text, Aspect, Valence, Arousal]
    """
    if "Quadruplet" in data[0]:
        df = pd.json_normalize(data, "Quadruplet", ["ID", "Text"])
        df[["Valence", "Arousal"]] = df["VA"].str.split("#", expand=True).astype(float)
        df.drop(columns=["VA", "Category", "Opinion"], inplace=True)

    elif "Triplet" in data[0]:
        df = pd.json_normalize(data, "Triplet", ["ID", "Text"])
        df[["Valence", "Arousal"]] = df["VA"].str.split("#", expand=True).astype(float)
        df.drop(columns=["VA", "Opinion"], inplace=True)

    elif "Aspect_VA" in data[0]:
        df = pd.json_normalize(data, "Aspect_VA", ["ID", "Text"])
        df.rename(columns={df.columns[0]: "Aspect"}, inplace=True)
        df[["Valence", "Arousal"]] = df["VA"].str.split("#", expand=True).astype(float)

    elif "Aspect" in data[0]:
        df = pd.json_normalize(data, "Aspect", ["ID", "Text"])
        df.rename(columns={df.columns[0]: "Aspect"}, inplace=True)
        df["Valence"], df["Arousal"] = 0, 0

    else:
        raise ValueError("Unknown JSONL structure; check dataset format.")

    df.drop_duplicates(subset=["ID", "Aspect"], inplace=True)
    return df


class VADataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.texts = df["Text"].tolist()
        self.aspects = df["Aspect"].tolist()
        self.labels = df[["Valence", "Arousal"]].values.astype(float)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): 
        return len(self.texts)

    def __getitem__(self, idx):
        text = f"{self.aspects[idx]}: {self.texts[idx]}"
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float)
        }
