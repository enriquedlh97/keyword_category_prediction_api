# Data loading and preprocessing
from modeling.baseline_models.preprocessing import get_and_preprocess_data, remove_alphanumeric, \
    process_punctuation_and_lower_cased, process_new_lines, remove_non_ascii
from modeling.baseline_models.training_and_testing import train_category, test_category
# Model
from sklearn.ensemble import RandomForestClassifier
# Vectorizers
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# Metrics
from sklearn.metrics import log_loss
# Stop words
from nltk.corpus import stopwords
# Other
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd


def model_evaluation(category, n_estimators, criterion, max_features, bootstrap, warm_start, class_weight, ccp_alpha,
                     vectorizer_selection, strip_accents, lowercase, ngram_range, stop_words, alphanumeric,
                     punctuation_and_lower_cased, new_lines, non_ascii, k_folds, verbose):
    # Set k-fold cross validation scheme
    cv = KFold(n_splits=k_folds, shuffle=True, random_state=69)

    # Data fetching and preprocessing, and basic set up
    pd_train_dict, label_columns = get_and_preprocess_data(train=True, test=False, sampling=0.05)
    pd_data = pd_train_dict[category]['data']

    # Preprocess data
    pd_data = apply_preprocessing(pd_data=pd_data,
                                  alphanumeric=alphanumeric,
                                  punctuation_and_lower_cased=punctuation_and_lower_cased,
                                  new_lines=new_lines,
                                  non_ascii=non_ascii)

    # For fold validation loss results
    fold_validation_results = []

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(cv.split(pd_data)):
        if verbose >= 1:
            # Print
            print(f'FOLD {fold + 1}')
            print('--------------------------------')

        # Split data and
        pd_train = pd_data.filter(items=train_ids, axis=0)
        pd_test = pd_data.filter(items=test_ids, axis=0)

        # Initialize model and vectorizer
        model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_features=max_features,
                                       bootstrap=bootstrap, warm_start=warm_start, class_weight=class_weight,
                                       ccp_alpha=ccp_alpha, n_jobs=-1)
        vectorizer = vectorizer_selection(strip_accents=strip_accents, lowercase=lowercase, ngram_range=ngram_range,
                                          stop_words=stop_words)

        model, vectorizer = train_category(pd_data=pd_train, category=category, model=model, vectorizer=vectorizer)

        y_pred, y_true = test_category(pd_data=pd_test, category=category, model=model, vectorizer=vectorizer,
                                       avg_precision=False, roc_auc=False, params_opt=True)

        fold_loss = log_loss(y_true=y_true, y_pred=y_pred)

        if verbose >= 2:
            print(f"Validation loss for fold {fold + 1}: {fold_loss}")
            print('--------------------------------')

        fold_validation_results.append(fold_loss)

    if verbose >= 1:
        print(f'Average validation loss: {np.array(fold_validation_results).mean()}',
              f'Std dev of validation loss: {np.array(fold_validation_results).std()}')

    return np.array(fold_validation_results).mean()


def evaluate_model(category, n_estimators, criterion, max_features, bootstrap, warm_start, class_weight, ccp_alpha,
                   vectorizer_selection, strip_accents, lowercase, ngram_range, english, italian, french, spanish,
                   dutch, romanian, danish, norwegian, german, swedish, portuguese, finnish, alphanumeric,
                   punctuation_and_lower_cased, new_lines, non_ascii, k_folds=5, verbose=1):
    # Fix model hyperparamters
    n_estimators = round(n_estimators)
    criterion = "gini" if criterion <= 0.5 else "entropy"
    max_features = get_max_features(max_features)
    bootstrap = True if bootstrap <= 0.5 else False
    warm_start = True if warm_start <= 0.5 else False
    class_weight = get_class_weight(class_weight)
    category = get_category(category)

    # Fix vectorizer parameters
    vectorizer_selection = get_vectorizer(vectorizer_selection)
    strip_accents = get_strip_accents(strip_accents)
    lowercase = True if lowercase <= 0.5 else False
    ngram_range = get_ngram_range(ngram_range)
    stop_words = get_stop_words(english, italian, french, spanish, dutch, romanian, danish, norwegian, german, swedish,
                                portuguese, finnish)

    # Preprocessing
    alphanumeric = remove_alphanumeric if alphanumeric <= 0.5 else None
    punctuation_and_lower_cased = process_punctuation_and_lower_cased if punctuation_and_lower_cased <= 0.5 else None
    new_lines = process_new_lines if new_lines <= 0.5 else None
    non_ascii = remove_non_ascii if non_ascii <= 0.5 else None

    cv_loss = model_evaluation(category, n_estimators, criterion, max_features, bootstrap, warm_start, class_weight,
                               ccp_alpha, vectorizer_selection, strip_accents, lowercase, ngram_range, stop_words,
                               alphanumeric, punctuation_and_lower_cased, new_lines, non_ascii, k_folds, verbose)

    return -cv_loss


def get_class_weight(class_weight):
    # set class_weight
    if class_weight <= 1.0 / 3:
        return 'balanced'
    elif 2.0 / 3 >= class_weight > 1.0 / 3:
        return 'balanced_subsample'
    elif 3.0 / 3 >= class_weight > 2.0 / 3:
        return None


def get_max_features(max_features):
    # set max_features
    if max_features <= 1.0 / 3:
        return 'sqrt'
    elif 2.0 / 3 >= max_features > 1.0 / 3:
        return 'log2'
    elif 3.0 / 3 >= max_features > 2.0 / 3:
        return None


def get_vectorizer(vectorizer_selection):
    if round(vectorizer_selection):
        return CountVectorizer
    else:
        return TfidfVectorizer


def get_category(category):
    category_list = [
        "Health",
        "Vehicles",
        "Hobbies & Leisure",
        "Food & Groceries",
        "Retailers & General Merchandise",
        "Arts & Entertainment",
        "Jobs & Education",
        "Law & Government",
        "Home & Garden",
        "Finance",
        "Computers & Consumer Electronics",
        "Internet & Telecom",
        "Sports & Fitness",
        "Dining & Nightlife",
        "Business & Industrial",
        "Occasions & Gifts",
        "Travel & Tourism",
        "News, Media & Publications",
        "Apparel",
        "Beauty & Personal Care",
        "Family & Community",
        "Real Estate"
    ]
    return category_list[round(category)]


def get_strip_accents(strip_accents):
    # set strip_accents
    if strip_accents <= 1.0 / 3:
        return None
    elif 2.0 / 3 >= strip_accents > 1.0 / 3:
        return "ascii"
    elif 3.0 / 3 >= strip_accents > 2.0 / 3:
        return "unicode"


def get_ngram_range(ngram_range):
    # set ngram_range
    if ngram_range <= 1.0 / 3:
        return 1, 1
    elif 2.0 / 3 >= ngram_range > 1.0 / 3:
        return 1, 2
    elif 3.0 / 3 >= ngram_range > 2.0 / 3:
        return 2, 2


def get_stop_words(english, italian, french, spanish, dutch, romanian, danish, norwegian, german, swedish,
                   portuguese, finnish):
    stopwords_list = set()

    languages = ['english', 'italian', 'french', 'spanish', 'dutch', 'romanian', 'danish',
                 'norwegian', 'german', 'swedish', 'portuguese', 'finnish']

    language_selection = [english, italian, french, spanish, dutch, romanian, danish, norwegian, german, swedish,
                          portuguese, finnish]

    for language, selection in zip(languages, language_selection):
        if round(selection):
            for word in stopwords.words(language):
                stopwords_list.add(word)

    if stopwords_list:
        return stopwords_list
    else:
        return None


def apply_preprocessing(pd_data, alphanumeric, punctuation_and_lower_cased, new_lines, non_ascii):
    if alphanumeric is not None:
        pd_data['keyword'] = pd_data['keyword'].map(alphanumeric)
    if punctuation_and_lower_cased is not None:
        pd_data['keyword'] = pd_data['keyword'].map(punctuation_and_lower_cased)
    if new_lines is not None:
        pd_data['keyword'] = pd_data['keyword'].map(new_lines)
    if non_ascii is not None:
        pd_data['keyword'] = pd_data['keyword'].map(non_ascii)
    return pd_data