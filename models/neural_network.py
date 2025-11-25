import numpy as np
from models.logistic_model import sigmoid

class NeuralNetworkCustom:
    def __init__(self, n_inputs, n_hidden=8, lr=0.01, epochs=2000):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.lr = lr
        self.epochs = epochs
        
        self.W1 = np.random.randn(n_inputs, n_hidden)*0.01
        self.b1 = np.zeros((1,n_hidden))
        self.W2 = np.random.randn(n_hidden,1)*0.01
        self.b2 = np.zeros((1,1))
    
    def forward(self, X):
        self.z1 = X@self.W1 + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = self.a1@self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def fit(self, X, y):
        m = X.shape[0]
        
        for _ in range(self.epochs):

            y_hat = self.forward(X)
            dz2 = y_hat - y
            dW2 = (1/m) * self.a1.T @ dz2
            db2 = (1/m) * dz2.sum(axis=0, keepdims=True)

            da1 = dz2 @ self.W2.T
            dz1 = da1 * (self.a1*(1-self.a1))
            dW1 = (1/m) * X.T @ dz1
            db1 = (1/m) * dz1.sum(axis=0, keepdims=True)

            self.W2 -= self.lr*dW2
            self.b2 -= self.lr*db2
            self.W1 -= self.lr*dW1
            self.b1 -= self.lr*db1

    def predict(self,X):
        return (self.forward(X)>=0.5).astype(int)

    def predict_proba(self, X):
        return self.forward(X)