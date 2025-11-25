import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

class LogisticRegressionCustom:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.loss_history = []

    def fit(self, X, y):
        m, n = X.shape
        self.w = np.zeros((n,1))
        self.b = 0

        for _ in range(self.n_iters):
            z = X @ self.w + self.b
            y_hat = sigmoid(z)

            dw = (1/m) * X.T @ (y_hat - y)
            db = (1/m) * np.sum(y_hat - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db
    
    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)