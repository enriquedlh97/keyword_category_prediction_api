from bayes_opt import BayesianOptimization
from modeling.baseline_models.hyperparameters.support_vector_machine import evaluate_model

# For saving progress
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

# Stop words
import nltk

# Others
import os
import argparse
import time
import json
import warnings

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--i', type=int, default=2, help="Set number of steps for random exploration")
parser.add_argument('--n', type=int, default=5, help="Set number of steps for bayesian optimization")
parser.add_argument('--c', type=int, default=0, help="Set category number. The categories go from 0 to 21 in the order as they are found in the config.json file")
parser.add_argument('--w', dest='w', action='store_false', default=True, help="True for ignoring warnings, False otherwise")
args = parser.parse_args()

# Download stopwords to avoid errors
nltk.download('stopwords')

if args.w is True:
    warnings.filterwarnings("ignore")

with open("config.json") as json_file:
    config = json.load(json_file)

category = config["CLASS_NAMES"][args.c]
print(f"Optimizing hyperparameters for {category}")

start_time = time.time()

if not os.path.exists('hyperparameter_optimization/support_vector_machine/logs'):
    os.makedirs('hyperparameter_optimization/support_vector_machine/logs')

# Set path to logs
path_to_logs = 'hyperparameter_optimization/support_vector_machine/logs'

logger = JSONLogger(path="{path}/logs_{category}.json".format(path=path_to_logs,
                                                              category=category.lower().replace(" ", "_")))

# Set exploration and exploitation parameters
init_points = args.i
n_iter = args.n

# Bounded region of parameter space
pbounds = {
    'category': (args.c, args.c + 0.1),
    'C': (1e-5, 100),
    'max_iter': (10, 10000),
    'class_weight': (0, 1),
    'vectorizer_selection': (0, 1),
    'strip_accents': (0, 1),
    'lowercase': (0, 1),
    'ngram_range': (0, 1),
    'english': (0, 1),
    'italian': (0, 1),
    'french': (0, 1),
    'spanish': (0, 1),
    'dutch': (0, 1),
    'romanian': (0, 1),
    'danish': (0, 1),
    'norwegian': (0, 1),
    'german': (0, 1),
    'swedish': (0, 1),
    'portuguese': (0, 1),
    'finnish': (0, 1),
    'alphanumeric': (0, 1),
    'punctuation_and_lower_cased': (0, 1),
    'new_lines': (0, 1),
    'non_ascii': (0, 1),
}

# Initialize optimizer
optimizer = BayesianOptimization(
    f=evaluate_model,
    pbounds=pbounds,
    verbose=2,  # verbose=1 prints only when a max is observed
    random_state=1,
)

# Sets up logger
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

# Optimize
optimizer.maximize(
    init_points=init_points,
    n_iter=n_iter,
)

print(optimizer.max, flush=True)

print("--- %s seconds ---" % (time.time() - start_time), flush=True)
