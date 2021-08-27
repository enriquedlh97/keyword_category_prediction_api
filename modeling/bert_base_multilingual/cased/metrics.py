import numpy as np
from torchmetrics import AveragePrecision
from pytorch_lightning.metrics.functional import auroc


def mean_auc_roc(predictions, labels, label_columns):
    auc_roc_values = []
    auc_roc_dict = {}

    print("Average precision per tag")
    for i, name in enumerate(label_columns):
        auc_roc = auroc(predictions[:, i], labels[:, i])
        auc_roc_dict[name] = auc_roc.reshape(-1).numpy()[0]
        auc_roc_values.append(auc_roc.reshape(-1).numpy()[0])

    return np.array(auc_roc_values).mean(), auc_roc_dict


def mean_avg_precision(predictions, labels, label_columns):
    avg_precision_values = []
    avg_precision_dict = {}
    average_precision = AveragePrecision(pos_label=1)

    print("Average precision per tag")
    for i, name in enumerate(label_columns):
        avg_precision = average_precision(predictions[:, i], labels[:, i])
        avg_precision_dict[name] = avg_precision.reshape(-1).numpy()[0]
        avg_precision_values.append(avg_precision.reshape(-1).numpy()[0])

    return np.array(avg_precision_values).mean(), avg_precision_dict
