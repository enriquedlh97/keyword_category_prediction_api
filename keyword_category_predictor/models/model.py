import json
import torch
# from torch import nn
from transformers import BertTokenizer
# from .bert_multilingual import KeywordCategorizer
from keyword_category_predictor.models.bert_multilingual import KeywordCategorizer

with open("config.json") as json_file:
    config = json.load(json_file)


class Model:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(config["MODEL"])
        category_predictor = KeywordCategorizer(len(config["CLASS_NAMES"]))
        category_predictor.load_from_checkpoint(
            config["PRETRAINED_MODEL"],
            n_classes=len(config["CLASS_NAMES"])
        )
        category_predictor.eval()
        category_predictor.freeze()
        self.category_predictor = category_predictor.to(self.device)

    def predict(self, text):

        encoded_text = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=config["MAX_SEQUENCE_LEN"],
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoded_text["input_ids"].to(self.device)
        attention_mask = encoded_text["attention_mask"].to(self.device)

        confidence, probabilities = self.category_predictor(input_ids, attention_mask)
        probabilities = probabilities.flatten().numpy()

        return (
            confidence,
            dict(zip(config["CLASS_NAMES"], probabilities)),
        )


model = Model()


def get_model():
    return model
