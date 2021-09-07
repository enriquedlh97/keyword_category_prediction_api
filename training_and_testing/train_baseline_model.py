# Data loading and preprocessing
from modeling.baseline_models.preprocessing import get_and_preprocess_data
from modeling.baseline_models.training_and_testing import train_models, set_model_and_vectorizer_params, \
    save_trained_models
# Vectorizers
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# Models
from sklearn.linear_model import LogisticRegression
from modeling.baseline_models.training_and_testing import load_hyperparams
# Other
import time
import json
import warnings
import argparse


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--d', dest='d', action='store_false', default=True, help="Use default parameters for models and vectorizers")
parser.add_argument('--s', type=float, default=1, help="Define sampling proportion fo data")
parser.add_argument('--m', type=str, default='lr',
                    help="Set model to be trained. 'lr' for Linear Regression. 'svm' for Support Vector Machine. 'rf' for Random Forest")
parser.add_argument('--w', dest='w', action='store_false', default=True, help="True for ignoring warnings, False otherwise")

args = parser.parse_args()
if args.m == 'lr':
    model_name = 'Logistic Regression'
    hyperparams_path = 'logistic_regression'
elif args.m == 'svm':
    model_name = 'Support Vector Machine'
    hyperparams_path = 'support_vector_machine'
elif args.m == 'rf':
    model_name = 'Random Forest'
    hyperparams_path = 'random_forest'

default = args.d
sampling = args.s

if args.w is True:
    warnings.filterwarnings("ignore")

print(f"Training {model_name}...", flush=True)

# Get model and vectorizer hyperparameters
with open(f"training_and_testing/{hyperparams_path}/hyperparameters.json") as json_file:
    model_params = json.load(json_file)

# Model definition
if default is True:
    print('Using default model hyperparameters', flush=True)
    model_params_dict = model_params['DEFAULT']['MODEL']
    print('Using default vectorizer parameters', flush=True)
    vectorizer_params_dict = model_params['DEFAULT']['VECTORIZER']['PARAMS']
else:  # @TODO: add loading of optimized hyperparameters
    print('Using optimal model hyperparameters', flush=True)
    print('Using optimal vectorizer parameters', flush=True)

# Data fetching and preprocessing, and basic set up
pd_train_dict, pd_test_dict, label_columns = get_and_preprocess_data(train=True, test=True, sampling=sampling)
models_and_params = {model_name: pd_train_dict}
hyperparams = load_hyperparams(model_params_dict=model_params_dict,
                               vectorizer_params_dict=vectorizer_params_dict,
                               label_columns=label_columns,
                               default=default)

# Set model, model hyperparameters, vectorizer and vectorizer parameters
print('Setting up model', flush=True)
models_and_params = set_model_and_vectorizer_params(hyperparams=hyperparams,
                                                    models_and_params=models_and_params,
                                                    label_columns=label_columns,
                                                    model=LogisticRegression,
                                                    model_name=model_name,
                                                    vectorizer=eval(model_params['DEFAULT']['VECTORIZER']['NAME']))

# Train categories
start_time = time.time()
print('Training model', flush=True)
models_and_params = train_models(models_and_params=models_and_params, model_name=model_name)
print("The model trained in: %s seconds" % (time.time() - start_time), flush=True)

# Save models
print('Saving model', flush=True)
save_trained_models(models_and_params=models_and_params, label_columns=label_columns)
