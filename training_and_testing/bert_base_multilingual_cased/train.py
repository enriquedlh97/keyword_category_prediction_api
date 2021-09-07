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

args = parser.parse_args()
run_name = args.n

print(f"Model will be saved to: 'assets/bert_final_training/{run_name}'")

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
# Finetune for 4 epochs is recommended
MODEL_NAME = 'bert-base-multilingual-cased'
LABEL_COLUMNS = list(categories_dict.keys())
MAX_TOKEN_COUNT = args.t
N_EPOCHS = args.e
BATCH_SIZE = args.b  # batch sizes: 8, 16, 32, 64, 128
LEARNING_RATE = args.l  # learning rates: 3e-4, 1e-4, 5e-5, 3e-5, 2e-5
DROPOUT = args.d

# Optimizer scheduler
STEPS_PER_EPOCH = len(pd_train) // BATCH_SIZE
TOTAL_TRAINING_STEPS = STEPS_PER_EPOCH * N_EPOCHS
WARMUP_STEPS = TOTAL_TRAINING_STEPS // 5

# DATASET
print('Initializing dataset', flush=True)
data_module = KeywordDataModule(pd_train, pd_test, BertTokenizer.from_pretrained(MODEL_NAME), LABEL_COLUMNS, BATCH_SIZE,
                                MAX_TOKEN_COUNT)

# MODEL
print('Initializing model', flush=True)
model = KeywordCategorizer(len(LABEL_COLUMNS), LABEL_COLUMNS, TOTAL_TRAINING_STEPS, WARMUP_STEPS, MODEL_NAME,
                           LEARNING_RATE, DROPOUT)

# TRAINING

# Checkpoints and early stopping
print('Setting up model checkpoints, tensorboard logger and early stopping', flush=True)

if not os.path.exists('assets/bert_final_training'):
    os.makedirs('assets/bert_final_training')

if not os.path.exists(f"assets/bert_final_training/{run_name}"):
    os.makedirs(f"assets/bert_final_training/{run_name}")

checkpoint_callback = ModelCheckpoint(
    dirpath=f"assets/bert_final_training/{run_name}",
    filename="{epoch}-{val_loss:.5f}-best-checkpoint",
    save_top_k=-1,
    verbose=True,
    monitor="val_loss",
    mode="min"
)

# b64_l5e-5

logger = TensorBoardLogger(f"assets/bert_final_training/{run_name}/lightning_logs", name="keyword-categories")

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=N_EPOCHS)

# Initialize trainer - Requires GPU

print('Initializing trainer', flush=True)
trainer = pl.Trainer(
    logger=logger,
    checkpoint_callback=True,
    callbacks=[checkpoint_callback, early_stopping_callback],
    max_epochs=N_EPOCHS,
    gpus=1,  # If no GPU available comment this line
    progress_bar_refresh_rate=10
)

# Start training_and_testing
print('Training model', flush=True)
start_time = time.time()
trainer.fit(model, data_module)
print("The model trained in: %s seconds" % (time.time() - start_time), flush=True)
