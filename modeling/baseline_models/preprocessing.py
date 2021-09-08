# Data reading and preprocessing
from modeling.bert_base_multilingual.cased.preprocessing import get_data
import re
import string


def build_category_datasets(pd_data, label_columns, train=True):
    """ Separates dataset into sections for each category

    :param label_columns:
    :param pd_data:
    :return:
    """

    category_datasets_dict = {}

    # Builds dictionary with all dataframes
    for category in label_columns:
        if train:
            category_datasets_dict[category] = {
                'data': pd_data.reset_index().loc[:, ['index', 'keyword', category]],
                'model': None,
                'vectorizer': None
            }
        else:
            category_datasets_dict[category] = pd_data.reset_index().loc[:, ['index', 'keyword', category]]

    return category_datasets_dict


def remove_alphanumeric(text):
    """ Removes numbers that have letters attached

    :return: function instance for removing numbers that have letters attached
    """
    return re.sub('\w*\d\w*', ' ', text)


def process_punctuation_and_lower_cased(text):
    """ Sets all strings to lower case and replaces punctuation with white space

    :return: function instance for setting all strings to lower case and replaces punctuation with white space
    """
    return re.sub('[%s]' % re.escape(string.punctuation), ' ', text.lower())


def process_new_lines(text):
    """ Replaces new lines (\n) with white spaces

    :return: function instance for replacing new lines (\n) with white spaces
    """
    return re.sub("\n", " ", text)


def remove_non_ascii(text):
    """ Removes non-ascii characters

    :return: function instance for removing non-ascii characters
    """
    return re.sub(r'[^\x00-\x7f]', r' ', text)


def get_and_preprocess_data(train=True, test=True,
                            train_path="dataset/keyword_categories/keyword_categories/keyword_categories.train.jsonl",
                            test_path="dataset/keyword_categories/keyword_categories/keyword_categories.test.jsonl",
                            sampling=1):

    pd_train, pd_test, label_columns = get_data(train=True, test=True, train_path=train_path, test_path=test_path,
                                                sampling=sampling)

    pd_train_dict = build_category_datasets(pd_train, label_columns)
    pd_test_dict = build_category_datasets(pd_test, label_columns, train=False)

    if train and test:
        return pd_train_dict, pd_test_dict, label_columns
    elif train and test is False:
        return pd_train_dict, label_columns
    elif train is False and test:
        return pd_test_dict, label_columns
