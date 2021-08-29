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
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--t', type=int, default=40, help="Set max token count")
parser.add_argument('--e', type=int, default=35, help="Set number of epochs")
parser.add_argument('--b', type=int, default=64, help="Set batch size")
parser.add_argument('--l', type=float, default=2e-5, help="Set learning rate")
parser.add_argument('--d', type=float, default=0.12, help="Set dropout rate")
args = parser.parse_args()

start_time = time.time()

# DATA

# Get data
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
pd_train = pd_train.sample(round(pd_train.shape[0] * .01))
pd_test = pd_test.sample(round(pd_test.shape[0] * .01))

# GLOBAL VARIABLES AND PARAMETERS

# Finetune for 4 epochs is recommended
MODEL_NAME = 'bert-base-multilingual-cased'
LABEL_COLUMNS = list(categories_dict.keys())
MAX_TOKEN_COUNT = args.t # 40
N_EPOCHS = args.e #1
BATCH_SIZE = args.b #64  # batch sizes: 8, 16, 32, 64, 128
LEARNING_RATE = args.l #2e-5  # learning rates: 3e-4, 1e-4, 5e-5, 3e-5, 2e-5
DROPOUT = args.d #.12

# Optimizer scheduler
STEPS_PER_EPOCH = len(pd_train) // BATCH_SIZE
TOTAL_TRAINING_STEPS = STEPS_PER_EPOCH * N_EPOCHS
WARMUP_STEPS = TOTAL_TRAINING_STEPS // 5

# DATASET

data_module = KeywordDataModule(pd_train, pd_test, BertTokenizer.from_pretrained(MODEL_NAME), LABEL_COLUMNS, BATCH_SIZE,
                                MAX_TOKEN_COUNT)

# MODEL

model = KeywordCategorizer(len(LABEL_COLUMNS), LABEL_COLUMNS, TOTAL_TRAINING_STEPS, WARMUP_STEPS, MODEL_NAME,
                           LEARNING_RATE, DROPOUT)

# TRAINING

# Checkpoints and early stopping

checkpoint_callback = ModelCheckpoint(
    dirpath="assets",
    #filename="dropout/{epoch}-{val_loss:.5f}-best-checkpoint",
    filename="dropout/best-checkpoint",
    save_top_k=-1,
    verbose=True,
    monitor="val_loss",
    mode="min"
)

logger = TensorBoardLogger("lightning_logs", name="keyword-categories")

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=35)

# Initialize trainer - Requires GPU

trainer = pl.Trainer(
    logger=logger,
    checkpoint_callback=True,
    callbacks=[checkpoint_callback, early_stopping_callback],
    max_epochs=N_EPOCHS,
    gpus=1,  # If no GPU available comment this line
    progress_bar_refresh_rate=10
)

# Start training

trainer.fit(model, data_module)

# TESTING

trainer.test()

# EVALUATION

# Load model
trained_model = KeywordCategorizer.load_from_checkpoint(
    'assets/dropout/best-checkpoint.ckpt',
    n_classes=len(LABEL_COLUMNS),
    label_columns=LABEL_COLUMNS
)

trained_model.eval()
trained_model.freeze()

# Send model to available device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trained_model = trained_model.to(device)

# Get evaluation dataset
test_dataset = KeywordDataset(pd_test, BertTokenizer.from_pretrained(MODEL_NAME), LABEL_COLUMNS, MAX_TOKEN_COUNT)

# Start evaluation
predictions = []
labels = []

for item in tqdm(test_dataset):
    _, prediction = trained_model(
        item["input_ids"].unsqueeze(dim=0).to(device),
        item["attention_mask"].unsqueeze(dim=0).to(device)
    )
    predictions.append(prediction.flatten())
    labels.append(item["labels"].int())

predictions = torch.stack(predictions).detach().cpu()
labels = torch.stack(labels).detach().cpu()

# COMPUTE METRICS

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

print("--- %s seconds ---" % (time.time() - start_time), flush=True)
