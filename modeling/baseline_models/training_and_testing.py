from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd


def train_category(pd_data, category, model, vectorizer):
    x_train, y_train = pd_data.keyword, pd_data[category]

    x_train_matrix = vectorizer.fit_transform(x_train)

    return model.fit(x_train_matrix, y_train)


def test_category(pd_data, category, model, vectorizer, avg_precision=True, roc_auc=False):
    x_test, y_test = pd_data.keyword, pd_data[category]

    x_test_matrix = vectorizer.transform(x_test)

    if avg_precision:
        return {'Average precision': [average_precision_score(model.predict(x_test_matrix), y_test)]}
    elif roc_auc:
        return {'AUC ROC': [roc_auc_score(model.predict(x_test_matrix), y_test)]}


def train_models(pd_data, models, vectorizers, label_columns):
    trained_models = []

    for category, model, vectorizer in zip(label_columns, models, vectorizers):
        trained_model = train_category(pd_data=pd_data, category=category, model=model[1], vectorizer=vectorizer)
        trained_models.append([model[0], trained_model])

    return trained_models


def test_models(pd_data, models, vectorizers, label_columns):
    pd_avg_precision_results, pd_auc_roc_results = pd.DataFrame(), pd.DataFrame()

    for category, model, vectorizer in zip(label_columns, models, vectorizers):
        avg_precision_score_data = test_category(pd_data=pd_data, category=category, model=model[1],
                                                 vectorizer=vectorizer, avg_precision=True, roc_auc=False)
        pd_avg_precision_results[category] = pd.DataFrame(avg_precision_score_data, index=[model[0]])
        pd_avg_precision_results.rename(columns={'Average precision': f"Average precision - {category}"}, inplace=True)

        auc_roc_score_data = test_category(pd_data=pd_data, category=category, model=model[1],
                                           vectorizer=vectorizer, avg_precision=False, roc_auc=True)
        pd_auc_roc_results[category] = pd.DataFrame(auc_roc_score_data, index=[model[0]])
        pd_auc_roc_results.rename(columns={'Average precision': f"Average precision - {category}"}, inplace=True)

    return pd_avg_precision_results.transpose(), pd_auc_roc_results.transpose()
