from sklearn.metrics import roc_auc_score, average_precision_score
from modeling.baseline_models.preprocessing import get_and_preprocess_data
import pandas as pd
import joblib
import os


def train_category(pd_data, category, model, vectorizer):
    x_train, y_train = pd_data.keyword, pd_data[category]

    x_train_matrix = vectorizer.fit_transform(x_train)

    return model.fit(x_train_matrix, y_train), vectorizer


def test_category(pd_data, category, model, vectorizer, avg_precision=True, roc_auc=False):
    x_test, y_test = pd_data.keyword, pd_data[category]

    x_test_matrix = vectorizer.transform(x_test)

    if avg_precision:
        return {'Average precision': [average_precision_score(model.predict(x_test_matrix), y_test)]}
    elif roc_auc:
        return {'AUC ROC': [roc_auc_score(model.predict(x_test_matrix), y_test)]}


def train_models(models_and_params, model_name, verbose=1):
    for category in models_and_params[model_name]:
        models_and_params[model_name][category]['model'], \
        models_and_params[model_name][category]['vectorizer'] = train_category(
            pd_data=models_and_params[model_name][category]['data'],
            category=category,
            model=models_and_params[model_name][category]['model'],
            vectorizer=models_and_params[model_name][category]['vectorizer']
        )

        if verbose >= 1:
            print(f"Model: {model_name}, Category: {category} - Training done")

    return models_and_params


def test_models(pd_data, model_name, label_columns, models_and_params=None, verbose=1):
    pd_avg_precision_results, pd_auc_roc_results = pd.DataFrame(columns=[label_columns], index=[model_name]), \
                                                   pd.DataFrame(columns=[label_columns], index=[model_name])

    for category in models_and_params[model_name]:
        avg_precision_score_data = test_category(pd_data=pd_data[category],
                                                 category=category,
                                                 model=models_and_params[model_name][category]['model'],
                                                 vectorizer=models_and_params[model_name][category]['vectorizer'],
                                                 avg_precision=True, roc_auc=False)
        pd_avg_precision_results[[category]] = avg_precision_score_data['Average precision'][0]
        pd_avg_precision_results.rename(columns={category: f"Average precision - {category}"}, inplace=True)

        auc_roc_score_data = test_category(pd_data=pd_data[category],
                                           category=category,
                                           model=models_and_params[model_name][category]['model'],
                                           vectorizer=models_and_params[model_name][category]['vectorizer'],
                                           avg_precision=False, roc_auc=True)
        pd_auc_roc_results[[category]] = auc_roc_score_data['AUC ROC'][0]
        pd_auc_roc_results.rename(columns={'Health': f"AUC ROC - {category}"}, inplace=True)

        if verbose >= 1:
            print(f"Model: {model_name}, Category: {category} - Testing done")

    return pd_avg_precision_results.transpose(), pd_auc_roc_results.transpose()


def build_dummy_dict(model_name, model_path='assets/logistic_regression'):
    pd_train_dict, pd_test_dict, label_columns = get_and_preprocess_data(train=True, test=True, sampling=0.001)
    models_and_params = {model_name: pd_train_dict}
    models_and_params = set_model_and_vectorizer_params(hyperparams=None, models_and_params=models_and_params,
                                                        label_columns=label_columns, model=None,
                                                        model_name=model_name, vectorizer=None, model_path=model_path)

    return models_and_params


def set_model_and_vectorizer_params(hyperparams, models_and_params, label_columns, model, model_name, vectorizer,
                                    model_path=None):

    for category in label_columns:
        if hyperparams is not None:
            # Initialize model with hyperparameters
            initialized_model = model(**hyperparams[category]['model'])
            # Initialize vectorizer with parameters
            initialized_vectorizer = vectorizer(**hyperparams[category]['vectorizer'])

            models_and_params[model_name][category]['model'] = initialized_model
            models_and_params[model_name][category]['vectorizer'] = initialized_vectorizer
        else:  # Load models
            models_and_params[model_name][category]['model'], \
            models_and_params[model_name][category]['vectorizer'] = load_trained_models(model_path, category)

    return models_and_params


def save_trained_models(models_and_params, label_columns):

    if not os.path.exists('assets'):
        os.makedirs('assets')

    model_name = list(models_and_params.keys())[0].lower().replace(" ", "_")

    if not os.path.exists(f"assets/{model_name}"):
        os.makedirs(f"assets/{model_name}")

    for category in label_columns:
        category_name = category.lower().replace(" ", "_")
        if not os.path.exists(f"assets/{model_name}/{category_name}"):
            os.makedirs(f"assets/{model_name}/{category_name}")

        if not os.path.exists(f"assets/{model_name}/{category_name}/model"):
            os.makedirs(f"assets/{model_name}/{category_name}/model")

        if not os.path.exists(f"assets/{model_name}/{category_name}/vectorizer"):
            os.makedirs(f"assets/{model_name}/{category_name}/vectorizer")

        # Save models
        joblib.dump(models_and_params[list(models_and_params.keys())[0]][category]['model'],
                    f"assets/{model_name}/{category_name}/model/model.sav")
        print(f"Model '{model_name}' for category '{category}' saved", flush=True)
        joblib.dump(models_and_params[list(models_and_params.keys())[0]][category]['vectorizer'],
                    f"assets/{model_name}/{category_name}/vectorizer/vectorizer.sav")
        print(f"Vectorizer for model '{model_name}' for category '{category}' saved", flush=True)

    return None


def load_trained_models(model_path='assets/logistic_regression', category=None, model=True, vectorizer=True):
    trained_model = joblib.load("".join([model_path, '/', category.lower().replace(" ", "_"), '/model/model.sav']))
    trained_vectorizer = joblib.load("".join([model_path, '/', category.lower().replace(" ", "_"),
                                      '/vectorizer/vectorizer.sav']))
    if model and vectorizer:
        return trained_model, trained_vectorizer
    if not model and vectorizer:
        return trained_vectorizer
    if not vectorizer and model:
        return trained_model
