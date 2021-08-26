import pandas as pd


def get_train_test_data(train=True, test=True,
                        train_path="dataset/keyword_categories/keyword_categories/keyword_categories.train.jsonl",
                        test_path="dataset/keyword_categories/keyword_categories/keyword_categories.test.jsonl"):
                        # train_path="drive/MyDrive/graphite/dataset/keyword_categories/keyword_categories/keyword_categories.train.jsonl",
                        # test_path="drive/MyDrive/graphite/dataset/keyword_categories/keyword_categories/keyword_categories.test.jsonl"):
    if train:
        pd_train = pd.read_json(path_or_buf=train_path, lines=True)
    if test:
        pd_test = pd.read_json(path_or_buf=test_path, lines=True)

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
