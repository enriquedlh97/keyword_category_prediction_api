import pytorch_lightning as pl
from torch.utils.data import DataLoader
from modeling.bert_base_multilingual.cased.text_dataset import KeywordDataset


class KeywordDataModule(pl.LightningDataModule):

    def __init__(self, pd_train, pd_test, tokenizer, label_columns: list, batch_size=8, max_token_len=128):
        super().__init__()
        self.batch_size = batch_size
        self.pd_train = pd_train
        self.pd_test = pd_test
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.label_columns = label_columns

    def setup(self, stage=None):

        self.train_dataset = KeywordDataset(
            self.pd_train,
            self.tokenizer,
            self.label_columns,
            self.max_token_len
        )
        self.test_dataset = KeywordDataset(
            self.pd_test,
            self.tokenizer,
            self.label_columns,
            self.max_token_len
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=2)
