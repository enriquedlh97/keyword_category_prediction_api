import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup
from torchmetrics import AveragePrecision
from pytorch_lightning.metrics.functional import auroc


class KeywordCategorizer(pl.LightningModule):

    def __init__(self, n_classes: int, label_columns: list, n_training_steps=None, n_warmup_steps=None,
                 model_name='bert-base-multilingual-cased'):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name, return_dict=True)
        # self.dropout = nn.Dropout(.12)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = nn.BCELoss()
        self.label_columns = label_columns

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        # output = self.dropout(output)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def training_epoch_end(self, outputs):
        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)
        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)
        # Compute ROC AUC
        roc_auc = []
        for i, name in enumerate(self.label_columns):
            class_roc_auc = auroc(predictions[:, i], labels[:, i])
            self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch)
            roc_auc.append(class_roc_auc.reshape(-1).numpy()[0])
        # Compute Mean ROC AUC
        self.logger.experiment.add_scalar(f"epoch_mean_roc_auc/Train", np.array(roc_auc).mean(), self.current_epoch)
        # Compute Average Precision
        avg_precision = []
        average_precision = AveragePrecision(pos_label=1)
        for i, name in enumerate(self.label_columns):
            class_avg_precision = average_precision(predictions[:, i], labels[:, i])
            self.logger.experiment.add_scalar(f"{name}_avg_precision/Train", class_avg_precision, self.current_epoch)
            avg_precision.append(class_avg_precision.reshape(-1).numpy()[0])
        # Compute Mean Average Precision
        self.logger.experiment.add_scalar(f"epoch_avg_precision/Train", np.array(class_avg_precision).mean(),
                                          self.current_epoch)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps
        )
        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )
