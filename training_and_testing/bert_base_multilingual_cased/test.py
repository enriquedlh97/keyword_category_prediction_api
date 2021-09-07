# Data reading and preprocessing
from modeling.bert_base_multilingual.cased.preprocessing import get_data
# Datset
from modeling.bert_base_multilingual.cased.text_dataset import KeywordDataset
# Model
from modeling.bert_base_multilingual.cased.model import KeywordCategorizer
# Transformer imports
from transformers import BertTokenizer
# Metrics
from modeling.bert_base_multilingual.cased.metrics import mean_auc_roc, mean_avg_precision, build_pd_results
# General
import torch
from tqdm.auto import tqdm
import time
import argparse
import warnings
import json

parser = argparse.ArgumentParser()
parser.add_argument('--s', type=float, default=1, help="Define sampling proportion for data")
parser.add_argument('--w', dest='w', action='store_false',
                    default=True, help="True for ignoring warnings, False otherwise")
parser.add_argument('--n', type=str, help="Set name of training execution where all model data is")
parser.add_argument('--p', type=str, help="Set path to model .ckpt file to be tested.")
parser.add_argument('--v', type=int, default=0, help="Set verbosity. 1 prints metrics at the end. 0 prints nothing")

args = parser.parse_args()
model_path = args.n
model_name = args.p

print(f"Test results will be saved to: 'assets/bert_final_training/{model_path}'")

if args.w is True:
    warnings.filterwarnings("ignore")

# Get data
print('Loading and preprocessing test data', flush=True)
pd_test, label_columns = get_data(train=False, test=True, sampling=args.s)

# GLOBAL VARIABLES AND PARAMETERS
print('Setting hyperparameters', flush=True)
file = open(f"assets/bert_final_training/{model_path}/hyperparams.json", "r")
hyperparams = json.load(file)

MODEL_NAME = 'bert-base-multilingual-cased'
LABEL_COLUMNS = label_columns
MAX_TOKEN_COUNT = hyperparams['MAX_TOKEN_COUNT']

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
auc_roc_results = build_pd_results(dictionary=auc_roc_class, metric="AUC ROC")
if args.v == 1:
    print("Mean AUC ROC:", mean_aucroc, "\n\nAUC ROC per category:", auc_roc_class, flush=True)

# Compute Mean Average Precision Average precision

mean_avg_prec, avg_prec_class = mean_avg_precision(predictions, labels, LABEL_COLUMNS)
avg_precision_results = build_pd_results(dictionary=avg_prec_class, metric="Average precision")
if args.v == 1:
    print("Mean Average Precision:", mean_avg_prec, "\n\nAverage Precision per category:", avg_prec_class, flush=True)

# Save results
print("Saving results", flush=True)
auc_roc_results.to_csv(f"assets/bert_final_training/{model_path}/auc_roc_results.csv")
avg_precision_results.to_csv(f"assets/bert_final_training/{model_path}/avg_precision_results.csv")
