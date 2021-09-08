from bayes_opt import BayesianOptimization
from modeling.baseline_models.hyperparameters.logistic_regression import evaluate_model

# For saving progress
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

# Others
import os
import argparse
import time

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--i', type=int, default=2, help="Set number of steps for random exploration")
parser.add_argument('--n', type=int, default=5, help="Set number of steps for bayesian optimization")
args = parser.parse_args()

start_time = time.time()

if not os.path.exists('hyperparameter_optimization/logistic_regression/logs'):
    os.makedirs('hyperparameter_optimization/logistic_regression/logs')

# Set path to logs
path_to_logs = 'hyperparameter_optimization/bert_base_multilingual_cased/logs'

logger = JSONLogger(path="{path}/logs.json".format(path=path_to_logs))

# Set exploration and exploitation parameters
init_points = args.i
n_iter = args.n

# Bounded region of parameter space
pbounds = {
    'category': (0, 0.1),  # Health
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
