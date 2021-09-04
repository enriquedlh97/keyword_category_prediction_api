# MAX_TOKEN_COUNT
# N_EPOCHS
# BATCH_SIZE
# LEARNING_RATE
# DROPOUT
# LEARNING_RATE_SCHEDULE
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
# Transformer learning rate schedulers
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup, \
    get_cosine_with_hard_restarts_schedule_with_warmup, get_linear_schedule_with_warmup, \
    get_polynomial_decay_schedule_with_warmup
# Logging and saving
import pytorch_lightning as pl
#from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
#from pytorch_lightning.loggers import TensorBoardLogger
# Other
from sklearn.model_selection import StratifiedKFold
from tqdm.notebook import tqdm


def model_evaluation(max_token_count, epochs, batch_size, learning_rate, dropout, learning_rate_schedule, k_folds,
                     verbose):
    # Set k-fold cross validation scheme
    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=69)

    # Get data
    pd_train = get_train_test_data(
        train=True,
        test=False
    )

    # Get categories
    categories_dict = get_categories(pd_train, pd_train)

    # Temporary sampling
    pd_train = pd_train.sample(round(pd_train.shape[0] * .01))

    # Add category columns and fill them
    pd_train = add_category_columns(pd_train, categories_dict)


    MODEL_NAME = 'bert-base-multilingual-cased'
    LABEL_COLUMNS = list(categories_dict.keys())

    #Hyperparameters
    MAX_TOKEN_COUNT = max_token_count
    N_EPOCHS = epochs
    BATCH_SIZE = batch_size
    LEARNING_RATE = learning_rate
    DROPOUT = dropout
    LEARNING_RATE_SCHEDULE = learning_rate_schedule

    # Optimizer scheduler
    STEPS_PER_EPOCH = len(pd_train) // BATCH_SIZE
    TOTAL_TRAINING_STEPS = STEPS_PER_EPOCH * N_EPOCHS
    WARMUP_STEPS = TOTAL_TRAINING_STEPS // 5




    # DATASET

    # data_module = KeywordDataModule(pd_train, pd_test, BertTokenizer.from_pretrained(MODEL_NAME), LABEL_COLUMNS,
    #                                 BATCH_SIZE,
    #                                 MAX_TOKEN_COUNT)

    # MODEL

    model = KeywordCategorizer(len(LABEL_COLUMNS), LABEL_COLUMNS, TOTAL_TRAINING_STEPS, WARMUP_STEPS, MODEL_NAME,
                               LEARNING_RATE, DROPOUT)

    # Start training

    # trainer.fit(model, data_module)

    # TESTING

    # trainer.test()


def evaluate_model(max_token_count, epochs, batch_size, learning_rate, dropout, learning_rate_schedule, k_folds=10,
                   verbose=1):
    # Fix hyperparameters
    max_token_count = round(max_token_count)
    epochs = round(epochs)
    batch_size = round(batch_size)
    learning_rate = round(learning_rate, 8)
    dropout = round(dropout, 5)
    learning_rate_schedule = get_schedule(learning_rate_schedule)

    cv_loss = model_evaluation(max_token_count, epochs, batch_size, learning_rate, dropout, learning_rate_schedule,
                               k_folds, verbose)

    return cv_loss


def get_schedule(schedule):

    # set schedule
    if schedule <= 1.0 / 5:
        schedule = get_constant_schedule_with_warmup
    elif 2.0 / 5 >= schedule > 1.0 / 5:
        schedule = get_cosine_schedule_with_warmup
    elif 3.0 / 5 >= schedule > 2.0 / 5:
        schedule = get_cosine_with_hard_restarts_schedule_with_warmup
    elif 4.0 / 5 >= schedule > 3.0 / 5:
        schedule = get_linear_schedule_with_warmup
    elif 5.0 / 5 >= schedule > 4.0 / 5:
        schedule = get_polynomial_decay_schedule_with_warmup

    return schedule


if __name__ == "__main__":
    max_token_count = 40
    epochs = 2
    batch_size = 64
    learning_rate = 2e-5
    dropout = 0.12
    learning_rate_schedule = 0.7  # Should select get_linear_schedule_with_warmup

    cv_loss = evaluate_model(max_token_count, epochs, batch_size, learning_rate, dropout, learning_rate_schedule)

    print(cv_loss)
