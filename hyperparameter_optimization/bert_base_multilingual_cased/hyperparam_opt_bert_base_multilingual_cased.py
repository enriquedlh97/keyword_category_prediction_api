from bayes_opt import BayesianOptimization
from modeling.bert_base_multilingual.cased.hyperparameters.optimization_structure import evaluate_model

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

if not os.path.exists('hyperparameter_optimization/bert_base_multilingual_cased/logs'):
    os.makedirs('hyperparameter_optimization/bert_base_multilingual_cased/logs')

# Set path to logs
path_to_logs = 'hyperparameter_optimization/bert_base_multilingual_cased/logs'

logger = JSONLogger(path="{path}/logs.json".format(path=path_to_logs))

# Set exploration and exploitation parameters
init_points = args.i
n_iter = args.n

# Bounded region of parameter space
pbounds = {
    'max_token_count': (33, 60),
    'epochs': (2, 2.1),
    'batch_size': (8, 128),
    'learning_rate': (5e-5, 1e-4),
    'dropout': (0, 1),
    'learning_rate_schedule': (0, 1),
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
