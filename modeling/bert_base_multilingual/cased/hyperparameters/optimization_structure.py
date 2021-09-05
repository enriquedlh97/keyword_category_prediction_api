# Preprocessing
from modeling.bert_base_multilingual.cased.preprocessing import get_train_test_data, get_categories, \
    add_category_columns
from modeling.bert_base_multilingual.cased.data_module import KeywordDataModule
# Model
from modeling.bert_base_multilingual.cased.model import KeywordCategorizer
# Transformer imports
from transformers import BertTokenizer
# Transformer learning rate schedulers
from transformers import get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup, \
    get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
# Logging and saving
import pytorch_lightning as pl
# Other
from sklearn.model_selection import KFold
import numpy as np


def model_evaluation(max_token_count, epochs, batch_size, learning_rate, dropout, learning_rate_schedule, k_folds,
                     verbose):
    # Set k-fold cross validation scheme
    cv = KFold(n_splits=k_folds, shuffle=True, random_state=69)

    # Get data
    pd_train = get_train_test_data(
        train=True,
        test=False
    )

    # Get categories
    categories_dict = get_categories(pd_train, pd_train)

    # Add category columns and fill them
    pd_train = add_category_columns(pd_train, categories_dict)

    # Temporary sampling
    pd_train = pd_train.sample(round(pd_train.shape[0] * .05))

    # Model details
    model_name = 'bert-base-multilingual-cased'
    label_columns = list(categories_dict.keys())

    # Optimizer scheduler
    steps_per_epoch = len(pd_train) // batch_size
    total_training_steps = steps_per_epoch * epochs
    warmup_steps = total_training_steps // 5

    # For fold validation loss results
    fold_validation_results = []

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(cv.split(pd_train)):
        if verbose >= 1:
            # Print
            print(f'FOLD {fold + 1} - epochs:{epochs}')
            print('--------------------------------')

        # DATASET
        data_module = KeywordDataModule(pd_train.filter(items=train_ids, axis=0),
                                        pd_train.filter(items=test_ids, axis=0),
                                        BertTokenizer.from_pretrained(model_name),
                                        label_columns,
                                        batch_size,
                                        max_token_count)

        # MODEL

        model = KeywordCategorizer(len(label_columns), label_columns, total_training_steps, warmup_steps,
                                   model_name, learning_rate, dropout, True, learning_rate_schedule)

        # Initialize trainer - Requires GPU

        trainer = pl.Trainer(
            logger=False,
            checkpoint_callback=False,
            max_epochs=epochs,
            gpus=1,  # If no GPU available comment this line
            progress_bar_refresh_rate=10
        )

        # TRAIN

        trainer.fit(model, data_module)

        # VALIDATE

        fold_validation_loss = trainer.validate(model, data_module)

        if verbose >= 2:
            # Print accuracy
            print(f"Validation loss for fold {fold + 1}: {fold_validation_loss[0]['val_loss']}%")
            print('--------------------------------')

        # Save validation loss for current fold
        fold_validation_results.append(fold_validation_loss[0]['val_loss'])

    if verbose >= 1:
        print(f'Average validation loss: {np.array(fold_validation_results).mean()} %',
              f'Std dev of validation loss: {np.array(fold_validation_results).std()} %')

    return np.array(fold_validation_results).mean()


def evaluate_model(max_token_count, epochs, batch_size, learning_rate, dropout, learning_rate_schedule, k_folds=2,
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

    return -cv_loss


def get_schedule(schedule):

    # set schedule
    if schedule <= 1.0 / 4:
        schedule = get_cosine_schedule_with_warmup
    elif 2.0 / 4 >= schedule > 1.0 / 4:
        schedule = get_cosine_with_hard_restarts_schedule_with_warmup
    elif 3.0 / 4 >= schedule > 2.0 / 4:
        schedule = get_linear_schedule_with_warmup
    elif 4.0 / 4 >= schedule > 3.0 / 4:
        schedule = get_polynomial_decay_schedule_with_warmup

    return schedule
