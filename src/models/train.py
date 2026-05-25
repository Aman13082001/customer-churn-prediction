"""Model training script."""

from sklearn.linear_model import LogisticRegression


def train(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model
