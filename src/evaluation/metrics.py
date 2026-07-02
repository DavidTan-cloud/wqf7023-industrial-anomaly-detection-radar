from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score
)

def evaluate(y_true,
             y_pred,
             anomaly_scores):

    return {
        "Accuracy":
            accuracy_score(y_true, y_pred),
        "Precision":
            precision_score(y_true, y_pred),
        "Recall":
            recall_score(y_true, y_pred),
        "F1":
            f1_score(y_true, y_pred),
        "AUC":
            roc_auc_score(y_true, anomaly_scores)
    }
