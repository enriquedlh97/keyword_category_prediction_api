import json
import torch
from torch import nn
from transformers import BertModel

with open("config.json") as json_file:
    config = json.load(json_file)


class KeywordCategorizer(nn.Module):
    def __init__(self, n_classes):
        super(KeywordCategorizer, self).__init__()
        self.bert = BertModel.from_pretrained(config["MODEL"])
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        # output = self.dropout(output)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output
