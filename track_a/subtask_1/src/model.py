import torch
import torch.nn as nn
from transformers import AutoModel

class TransformerVARegressor(nn.Module):
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(self.backbone.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        logits = self.regressor(self.dropout(pooled))

        # Map sigmoid output from [0,1] → [1,9]
        return torch.sigmoid(logits) * 8 + 1
