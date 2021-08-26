import pandas as pd
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset


class KeywordDataset(Dataset):

    def __init__(self, data: pd.DataFrame, tokenizer: BertTokenizer, label_columns: list, max_token_len: int = 128):
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_len
        self.label_columns = label_columns

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        keyword_text = data_row.keyword
        labels = data_row[self.label_columns]
        encoding = self.tokenizer.encode_plus(
            keyword_text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return dict(keyword_text=keyword_text, input_ids=encoding["input_ids"].flatten(),
                    attention_mask=encoding["attention_mask"].flatten(), labels=torch.FloatTensor(labels)
                    )
