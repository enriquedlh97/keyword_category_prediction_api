import json
import pandas as pd


def get_train_test_data(train=True, test=True,
                        train_path="dataset/keyword_categories/keyword_categories/keyword_categories.train.jsonl",
                        test_path="dataset/keyword_categories/keyword_categories/keyword_categories.test.jsonl"):
    if train:
        with open(train_path) as f:
            pd_train = pd.DataFrame([json.loads(l) for l in f.readlines()])
        # pd_train = pd.read_json(path_or_buf=train_path, lines=True)
    if test:
        with open(test_path) as f:
            pd_test = pd.DataFrame([json.loads(l) for l in f.readlines()])
        # pd_test = pd.read_json(path_or_buf=test_path, lines=True)

    if train and test:
        return pd_train, pd_test
    elif train and test is False:
        return pd_train
    elif train is False and test:
        return pd_test


def get_categories(pd_train=None, pd_test=None):
    categories_dict = {}

    # Get categories from train set
    if pd_train is not None:
        for row in pd_train['categories']:
            for category in row:
                if category not in categories_dict:
                    categories_dict[category] = 1

    # Get categories from test set
    if pd_test is not None:
        for row in pd_test['categories']:
            for category in row:
                if category not in categories_dict:
                    categories_dict[category] = 1

    return categories_dict


def add_category_columns(pd_data, categories_dict):
    # Add columns
    for category in categories_dict:
        pd_data[category] = 0

    # Fill columns
    for row in range(pd_data.shape[0] - 1):
        for category in pd_data["categories"].iloc[row]:
            pd_data.at[row, category] = 1

    return pd_data


def get_data(train=True, test=True,
             train_path="dataset/keyword_categories/keyword_categories/keyword_categories.train.jsonl",
             test_path="dataset/keyword_categories/keyword_categories/keyword_categories.test.jsonl",
             sampling=1):

    pd_train, pd_test = get_train_test_data(train=True, test=True, train_path=train_path, test_path=test_path)

    # Get categories
    categories_dict = get_categories(pd_train, pd_test)

    # Add category columns and fill them
    pd_train = add_category_columns(pd_train, categories_dict)
    pd_test = add_category_columns(pd_test, categories_dict)

    # Temporary sampling
    pd_train = pd_train.sample(round(pd_train.shape[0] * sampling))
    pd_test = pd_test.sample(round(pd_test.shape[0] * sampling))

    if train and test:
        return pd_train, pd_test
    elif train and test is False:
        return pd_train
    elif train is False and test:
        return pd_test
