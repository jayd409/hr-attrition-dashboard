import numpy as np
import pandas as pd

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

def train(X, y, lr=0.05, epochs=400):
    """Train logistic regression from scratch."""
    Xb = np.column_stack([np.ones(len(X)), X])
    theta = np.zeros(Xb.shape[1])

    for _ in range(epochs):
        h = sigmoid(Xb @ theta)
        theta -= lr * Xb.T @ (h - y) / len(y)

    return theta

def predict(X, theta):
    """Predict probability of attrition."""
    Xb = np.column_stack([np.ones(len(X)), X])
    return sigmoid(Xb @ theta)

def feature_importance(theta, names):
    """Return absolute coefficients as feature importance."""
    return pd.Series(np.abs(theta[1:]), index=names).sort_values(ascending=False)

def accuracy(y_true, y_pred):
    """Calculate classification accuracy."""
    y_binary = (y_pred >= 0.5).astype(int)
    return np.mean(y_true == y_binary)
