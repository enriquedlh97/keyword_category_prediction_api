from bayes_opt import BayesianOptimization
from modeling.baseline_models.hyperparameters.logistic_regression import evaluate_model

# For saving progress
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

# Others
import os
import argparse
import time
import json

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--i', type=int, default=2, help="Set number of steps for random exploration")
parser.add_argument('--n', type=int, default=5, help="Set number of steps for bayesian optimization")
parser.add_argument('--c', type=int, default=0, help="Set category number. The categories go from 0 to 21 in the order as they are found in the config.json file")
args = parser.parse_args()

with open("config.json") as json_file:
    config = json.load(json_file)

category = config["CLASS_NAMES"][args.c]
print(f"Optimizing hyperparameters for {category}")

start_time = time.time()

if not os.path.exists('hyperparameter_optimization/logistic_regression/logs'):
    os.makedirs('hyperparameter_optimization/logistic_regression/logs')

# Set path to logs
path_to_logs = 'hyperparameter_optimization/logistic_regression/logs'

logger = JSONLogger(path="{path}/logs_{category}.json".format(path=path_to_logs,
                                                              category=category.lower().replace(" ", "_")))

# Set exploration and exploitation parameters
init_points = args.i
n_iter = args.n

# Bounded region of parameter space
pbounds = {
    'category': (args.c, args.c + 0.1),  # Health
    'C': (1e-5, 100),
    'class_weight': (0, 1),
    'solver': (0, 1),
    'max_iter': (10, 10000),
    'warm_start': (0, 1),
    'vectorizer_selection': (0, 1),
    'strip_accents': (0, 1),
    'lowercase': (0, 1),
    'ngram_range': (0, 1),
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
