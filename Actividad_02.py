import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class LogisticNeuron:

    def __init__(self, n_inputs, learning_rate=0.1):
        self.w = - 1 + 2 * np.random.rand(n_inputs)
        self.b = - 1 + 2 * np.random.rand()
        self.eta = learning_rate

    def predict_proba(self, X):
        Z = np.dot(self.w, X) + self.b
        Y_est = 1 / (1 + np.exp(-Z))
        return Y_est

    def predict(self, X):
        Z = np.dot(self.w, X) + self.b
        Y_est = 1 / (1 + np.exp(-Z))
        return 1 * (Y_est > 0.5)

    def train(self, X, Y, epochs=1000):
        p = X.shape[1]
        for _ in range(epochs):
            Y_est = self.predict_proba(X)
            self.w += (self.eta / p) * np.dot((Y - Y_est), X.T).ravel()
            self.b += (self.eta / p) * np.sum(Y - Y_est)


df = pd.read_csv('./DataSets/cancer.csv')
X = np.asanyarray(df.drop(columns=['Class'])).T
Y = np.asanyarray(df[['Class']]).T.ravel()
print(X.shape)
print(Y.shape)

n, p = X.shape
for i in range(n):
    X[i, :] = (X[i, :] - X[i, :].min()) / (X[i, :].max() - X[i, :].min())

neuron = LogisticNeuron(9, 0.4)
neuron.train(X, Y, epochs=5000)

m = np.random.randint(p)
print(m)
print('Probabilidad: ')
print(neuron.predict_proba(X[:, m]))
print('Predicción: ')
print(neuron.predict(X[:, m]))
print('Valor Esperado: ')
print(Y[m])

Yest = np.zeros((p,))
for i in range(p):
    Yest[i] = neuron.predict(X[:, i])

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print('Accuracy: ', accuracy_score(Y, Yest))
print('Confusion matrix: \n', confusion_matrix(Y, Yest))
