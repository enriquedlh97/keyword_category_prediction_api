# Data reading and preprocessing
from modeling.bert_base_multilingual.cased.preprocessing import get_train_test_data, get_categories, \
    add_category_columns
# Datset
from modeling.bert_base_multilingual.cased.text_dataset import KeywordDataset
from modeling.bert_base_multilingual.cased.data_module import KeywordDataModule
# Model
from modeling.bert_base_multilingual.cased.model import KeywordCategorizer
# Transformer imports
from transformers import BertTokenizer
# Logging and saving
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
# Metrics
from modeling.bert_base_multilingual.cased.metrics import mean_auc_roc, mean_avg_precision
# General
import torch
from tqdm.auto import tqdm
from datetime import datetime
import time
import argparse
import warnings
import os

parser = argparse.ArgumentParser()
parser.add_argument('--t', type=int, default=40, help="Set max token count")
parser.add_argument('--e', type=int, default=20, help="Set number of epochs")
parser.add_argument('--b', type=int, default=64, help="Set batch size")
parser.add_argument('--l', type=float, default=2e-5, help="Set learning rate")
parser.add_argument('--d', type=float, default=0.12, help="Set dropout rate")
parser.add_argument('--s', type=float, default=1, help="Define sampling proportion for data")
parser.add_argument('--w', dest='w', action='store_false',
                    default=True, help="True for ignoring warnings, False otherwise")
parser.add_argument('--n', type=str, default=str(datetime.now()).lower().replace(" ", "_").replace(":", "-"),
                    help="Define name of training execution. If nothing is specified then the current datetime will be used")
parser.add_argument('--p', type=str, help="Set path of model to be tested")

args = parser.parse_args()
model_path = args.p
model_name = args.n

print(f"Test results will be saved to: 'assets/bert_final_training/{model_path}'")

if args.w is True:
    warnings.filterwarnings("ignore")

# Get data
print('Loading and preprocessing data', flush=True)
pd_train, pd_test = get_train_test_data(
    train_path="dataset/keyword_categories/keyword_categories/keyword_categories.train.jsonl",
    test_path="dataset/keyword_categories/keyword_categories/keyword_categories.test.jsonl"
)

# Get categories
categories_dict = get_categories(pd_train, pd_test)

# Add category columns and fill them
pd_train = add_category_columns(pd_train, categories_dict)
pd_test = add_category_columns(pd_test, categories_dict)

# Temporary sampling
pd_train = pd_train.sample(round(pd_train.shape[0] * args.s))
pd_test = pd_test.sample(round(pd_test.shape[0] * args.s))

# GLOBAL VARIABLES AND PARAMETERS
print('Setting hyperparameters', flush=True)

MODEL_NAME = 'bert-base-multilingual-cased'
LABEL_COLUMNS = list(categories_dict.keys())
MAX_TOKEN_COUNT = args.t
N_EPOCHS = args.e
BATCH_SIZE = args.b  # batch sizes: 8, 16, 32, 64, 128
LEARNING_RATE = args.l  # learning rates: 3e-4, 1e-4, 5e-5, 3e-5, 2e-5
DROPOUT = args.d

# Load model
print('Loading model', flush=True)
trained_model = KeywordCategorizer.load_from_checkpoint(
    f"assets/bert_final_training/{model_path}/{model_name}",
    n_classes=len(LABEL_COLUMNS),
    label_columns=LABEL_COLUMNS
)

trained_model.eval()
trained_model.freeze()

# Send model to available device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trained_model = trained_model.to(device)

# Get evaluation dataset
print('Building test dataset', flush=True)
test_dataset = KeywordDataset(pd_test, BertTokenizer.from_pretrained(MODEL_NAME), LABEL_COLUMNS, MAX_TOKEN_COUNT)

# Start evaluation
print('Testing', flush=True)
predictions = []
labels = []

start_time = time.time()
for item in tqdm(test_dataset):
    _, prediction = trained_model(
        item["input_ids"].unsqueeze(dim=0).to(device),
        item["attention_mask"].unsqueeze(dim=0).to(device)
    )
    predictions.append(prediction.flatten())
    labels.append(item["labels"].int())

predictions = torch.stack(predictions).detach().cpu()
labels = torch.stack(labels).detach().cpu()

print("The model was tested in %s seconds" % (time.time() - start_time), flush=True)

# COMPUTE METRICS
print('Computing metrics', flush=True)

# Compute AUC ROC metrics

mean_aucroc, auc_roc_class = mean_auc_roc(predictions, labels, LABEL_COLUMNS)

print("Mean AUC ROC:", mean_aucroc, "\n\nAUC ROC per category:", auc_roc_class, flush=True)

# Save AUC ROC metrics
file_object = open('final_metrics.txt', 'a')
file_object.write('Mean AUC ROC:{0}'.format(mean_aucroc))
file_object.write('\n\nAUC ROC per category:\n\n')
for key, value in auc_roc_class.items():
    file_object.write('%s:%s\n' % (key, value))

# Compute Mean Average Precision

mean_avg_prec, avg_prec_class = mean_avg_precision(predictions, labels, LABEL_COLUMNS)

print("Mean Average Precision:", mean_avg_prec, "\n\nAverage Precision per category:", avg_prec_class, flush=True)

# Save Mean Average Precision metrics
file_object.write('\n\n\n\nMean Average Precision:{0}'.format(mean_avg_prec))
file_object.write('\n\nAverage Precision per category::\n\n')
for key, value in avg_prec_class.items():
    file_object.write('%s:%s\n' % (key, value))
file_object.close()
