# Data reading and preprocessing
from modeling.bert_base_multilingual.cased.preprocessing import get_data
import re
import string


def build_category_datasets(pd_data):
    """ Separates dataset into sections for each category

    :param pd_data:
    :return:
    """

    # Builds dictionary with all dataframes
    category_datasets_dict = {
        'pd_health': pd_data.reset_index().loc[:, ['index', 'keyword', 'Health']],
        'pd_vehicles': pd_data.reset_index().loc[:, ['index', 'keyword', 'Vehicles']],
        'pd_hobbies': pd_data.reset_index().loc[:, ['index', 'keyword', 'Hobbies & Leisure']],
        'pd_food': pd_data.reset_index().loc[:, ['index', 'keyword', 'Food & Groceries']],
        'pd_retailers': pd_data.reset_index().loc[:, ['index', 'keyword', 'Retailers & General Merchandise']],
        'pd_arts': pd_data.reset_index().loc[:, ['index', 'keyword', 'Arts & Entertainment']],
        'pd_jobs': pd_data.reset_index().loc[:, ['index', 'keyword', 'Jobs & Education']],
        'pd_law': pd_data.reset_index().loc[:, ['index', 'keyword', 'Law & Government']],
        'pd_home': pd_data.reset_index().loc[:, ['index', 'keyword', 'Home & Garden']],
        'pd_finance': pd_data.reset_index().loc[:, ['index', 'keyword', 'Finance']],
        'pd_computers': pd_data.reset_index().loc[:, ['index', 'keyword', 'Computers & Consumer Electronics']],
        'pd_internet': pd_data.reset_index().loc[:, ['index', 'keyword', 'Internet & Telecom']],
        'pd_sports': pd_data.reset_index().loc[:, ['index', 'keyword', 'Sports & Fitness']],
        'pd_dining': pd_data.reset_index().loc[:, ['index', 'keyword', 'Dining & Nightlife']],
        'pd_business': pd_data.reset_index().loc[:, ['index', 'keyword', 'Business & Industrial']],
        'pd_gifts': pd_data.reset_index().loc[:, ['index', 'keyword', 'Occasions & Gifts']],
        'pd_travel': pd_data.reset_index().loc[:, ['index', 'keyword', 'Travel & Tourism']],
        'pd_news': pd_data.reset_index().loc[:, ['index', 'keyword', 'News, Media & Publications']],
        'pd_apparel': pd_data.reset_index().loc[:, ['index', 'keyword', 'Apparel']],
        'pd_beauty': pd_data.reset_index().loc[:, ['index', 'keyword', 'Beauty & Personal Care']],
        'pd_family': pd_data.reset_index().loc[:, ['index', 'keyword', 'Family & Community']],
        'pd_real_estate': pd_data.reset_index().loc[:, ['index', 'keyword', 'Real Estate']]
    }

    return category_datasets_dict


def remove_alphanumeric():
    """ Removes numbers that have letters attached

    :return: function instance for removing numbers that have letters attached
    """
    return lambda x: re.sub('\w*\d\w*', ' ', x)


def process_punctuation_and_lower_cased():
    """ Sets all strings to lower case and replaces punctuation with white space

    :return: function instance for setting all strings to lower case and replaces punctuation with white space
    """
    return lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())


def process_new_lines():
    """ Replaces new lines (\n) with white spaces

    :return: function instance for replacing new lines (\n) with white spaces
    """
    return lambda x: re.sub("\n", " ", x)


def remove_non_ascii():
    """ Removes non-ascii characters

    :return: function instance for removing non-ascii characters
    """
    return lambda x: re.sub(r'[^\x00-\x7f]', r' ', x)


def get_and_preprocess_data(train=True, test=True,
                            train_path="dataset/keyword_categories/keyword_categories/keyword_categories.train.jsonl",
                            test_path="dataset/keyword_categories/keyword_categories/keyword_categories.test.jsonl"):

    pd_train, pd_test = get_data(train=train, test=test, train_path=train_path, test_path=test_path)

    pd_train_dict = build_category_datasets(pd_train)
    pd_test_dict = build_category_datasets(pd_test)

    if train and test:
        return pd_train_dict, pd_test_dict
    elif train and test is False:
        return pd_train_dict
    elif train is False and test:
        return pd_test_dict
