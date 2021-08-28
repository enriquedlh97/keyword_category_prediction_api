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
MAX_TOKEN_COUNT = 40
N_EPOCHS = 10
BATCH_SIZE = 64  # batch sizes: 8, 16, 32, 64, 128
LEARNING_RATE = 2e-5  # learning rates: 3e-4, 1e-4, 5e-5, 3e-5, 2e-5

# Optimizer scheduler
STEPS_PER_EPOCH = len(pd_train) // BATCH_SIZE
TOTAL_TRAINING_STEPS = STEPS_PER_EPOCH * N_EPOCHS
WARMUP_STEPS = TOTAL_TRAINING_STEPS // 5

# DATASET

data_module = KeywordDataModule(pd_train, pd_test, BertTokenizer.from_pretrained(MODEL_NAME), LABEL_COLUMNS, BATCH_SIZE,
                                MAX_TOKEN_COUNT)

# MODEL

model = KeywordCategorizer(len(LABEL_COLUMNS), LABEL_COLUMNS, TOTAL_TRAINING_STEPS, WARMUP_STEPS, MODEL_NAME,
                           LEARNING_RATE)

# TRAINING

# Checkpoints and early stopping

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="best-checkpoint",
    save_top_k=1,
    verbose=True,
    monitor="val_loss",
    mode="min"
)

logger = TensorBoardLogger("lightning_logs", name="keyword-categories")

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)

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
    'checkpoints/best-checkpoint.ckpt',
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

# Compute Mean Average Precision

mean_avg_prec, avg_prec_class = mean_avg_precision(predictions, labels, LABEL_COLUMNS)

print("Mean Average Precision:", mean_avg_prec, "\n\nAverage Precision per category:", avg_prec_class, flush=True)

print("--- %s seconds ---" % (time.time() - start_time), flush=True)
