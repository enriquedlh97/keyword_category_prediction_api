from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd


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


def test_models(pd_data, models_and_params, model_name, label_columns, verbose=1):
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


def set_model_and_vectorizer_params(hyperparams, models_and_params, label_columns, model, model_name, vectorizer):
    for category in label_columns:
        # Initialize model with hyperparameters
        initialized_model = model(**hyperparams[category]['model'])
        # Initialize vectorizer with parameters
        initialized_vectorizer = vectorizer(**hyperparams[category]['vectorizer'])

        models_and_params[model_name][category]['model'] = initialized_model
        models_and_params[model_name][category]['vectorizer'] = initialized_vectorizer

    return models_and_params


def train_all_models():
    pass


def test_all_models():
    pass
