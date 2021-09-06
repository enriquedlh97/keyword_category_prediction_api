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


warnings.filterwarnings("ignore")


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--d', type=int, default=True, help="Use default parameters for models and vectorizers")
args = parser.parse_args()
default = args.d

# Get model and vectorizer hyperparameters
with open("training_and_testing/logistic_regression/hyperparameters.json") as json_file:
    lr_params = json.load(json_file)

# Model definition
model_name = 'Logistic Regression'
if default:
    lr_params_dict = lr_params['DEFAULT']['MODEL']
    lr_vectorizer_params_dict = lr_params['DEFAULT']['VECTORIZER']['PARAMS']
else:  # @TODO: add loading of optimized hyperparameters
    pass

# Data fetching and preprocessing, and basic set up
pd_train_dict, pd_test_dict, label_columns = get_and_preprocess_data(train=True, test=True, sampling=.05)
models_and_params = {model_name: pd_train_dict}
hyperparams = load_hyperparams(model_params_dict=lr_params_dict,
                               vectorizer_params_dict=lr_vectorizer_params_dict,
                               label_columns=label_columns,
                               default=default)

# Set model, model hyperparameters, vectorizer and vectorizer parameters
models_and_params = set_model_and_vectorizer_params(hyperparams=hyperparams,
                                                    models_and_params=models_and_params,
                                                    label_columns=label_columns,
                                                    model=LogisticRegression,
                                                    model_name=model_name,
                                                    vectorizer=eval(lr_params['DEFAULT']['VECTORIZER']['NAME']))

# Train categories
start_time = time.time()
models_and_params = train_models(models_and_params=models_and_params, model_name=model_name)

# Save models
save_trained_models(models_and_params=models_and_params, label_columns=label_columns)
