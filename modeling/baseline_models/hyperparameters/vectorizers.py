# Vectorizers
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# Stop words
from nltk.corpus import stopwords
import pandas as pd


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
