# Data loading and preprocessing
from modeling.baseline_models.preprocessing import get_and_preprocess_data
from modeling.baseline_models.training_and_testing import build_dummy_dict, test_models
# Other
import time
import json
import warnings
import argparse


# Parse arguments
parser = argparse.ArgumentParser()
# parser.add_argument('--d', dest='d', action='store_false', default=True, help="Use default parameters for models and vectorizers")
parser.add_argument('--s', type=float, default=1, help="Define sampling proportion for data")
parser.add_argument('--m', type=str, default='lr',
                    help="Set model to be trained. 'lr' for Linear Regression. 'svm' for Support Vector Machine. 'rf' for Random Forest")
parser.add_argument('--w', dest='w', action='store_false', default=True, help="True for ignoring warnings, False otherwise")

args = parser.parse_args()
if args.m == 'lr':
    model_name = 'Logistic Regression'
    # model = LogisticRegression
    model_path = 'logistic_regression'
elif args.m == 'svm':
    model_name = 'Support Vector Machine'
    # model = LinearSVC
    model_path = 'support_vector_machine'
elif args.m == 'rf':
    model_name = 'Random Forest'
    # model = RandomForestClassifier
    model_path = 'random_forest'

# default = args.d
sampling = args.s

if args.w is True:
    warnings.filterwarnings("ignore")

print(f"Testing {model_name}...", flush=True)

# Get data
print('Loading testing data', flush=True)
pd_test_dict, label_columns = get_and_preprocess_data(train=False, test=True, sampling=sampling)

# Load full model for testing
print('Loading model', flush=True)
models_and_params = build_dummy_dict(model_name=model_name, model_path=f"assets/{model_path}")

start_time = time.time()
print('Testing model', flush=True)
pd_avg_precision_results, pd_auc_roc_results = test_models(pd_data=pd_test_dict,
                                                                 models_and_params=models_and_params,
                                                                 model_name=model_name,
                                                                 label_columns=label_columns)
print("Testing the model took: %s seconds" % (time.time() - start_time), flush=True)

# Saving results
print('Saving results', flush=True)
pd_avg_precision_results.to_csv(f"assets/{model_path}/avg_precision_results.csv")
pd_auc_roc_results.to_csv(f"assets/{model_path}/auc_roc_results.csv")
