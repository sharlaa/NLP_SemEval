import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from data_loader import VADataset
from model import TransformerVARegressor
from utils import get_predictions, evaluate_predictions

def train_model(model_name, train_df, dev_df, epochs=30, lr=3e-5, batch_size=16):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(VADataset(train_df, tokenizer), batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(VADataset(dev_df, tokenizer), batch_size=batch_size)

    model = TransformerVARegressor(model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_val_loss = float('inf')
    patience, patience_counter = 7, 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/{epochs}"):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            preds = model(input_ids, attn_mask)
            loss = loss_fn(preds, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = eval_model(model, dev_loader, loss_fn, device)

        print(f"Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{model_name.replace('/', '_')}_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Final evaluation
    preds_v, preds_a, gold_v, gold_a = get_predictions(model, dev_loader, device)
    return evaluate_predictions(preds_a, preds_v, gold_a, gold_v)
