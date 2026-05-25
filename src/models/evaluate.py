"""Model evaluation utilities."""

from sklearn.metrics import accuracy_score


def evaluate(model, X, y):
    preds = model.predict(X)
    return accuracy_score(y, preds)
