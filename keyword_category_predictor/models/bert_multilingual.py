import json
import torch
import pytorch_lightning as pl
from torch import nn
from transformers import BertModel

with open("config.json") as json_file:
    config = json.load(json_file)


class KeywordCategorizer(pl.LightningModule):

    def __init__(self, n_classes: int):
        super().__init__()
        self.bert = BertModel.from_pretrained(config["MODEL"], return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.n_training_steps = 100
        self.n_warmup_steps = 100
        self.criterion = nn.BCELoss()
        self.dropout = nn.Dropout(0.12)
        self.label_columns = config["CLASS_NAMES"]
        self.learning_rate = 5e-5

        # Optimizer scheduler
        # STEPS_PER_EPOCH = len(pd_train) // BATCH_SIZE
        # TOTAL_TRAINING_STEPS = STEPS_PER_EPOCH * N_EPOCHS
        # WARMUP_STEPS = TOTAL_TRAINING_STEPS // 5

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        output.pooler_output = self.dropout(output.pooler_output)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output
