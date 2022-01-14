import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, n_inputs, learning_rate):
        self.w = - 1 + 2 * np.random.rand(n_inputs)
        self.b = - 1 + 2 * np.random.rand()
        self.eta = learning_rate

    def predict(self, X):
        _, p = X.shape
        y_est = np.zeros(p)
        for i in range(p):
            y_est[i] = np.dot(self.w, X[:, i]) + self.b
            if y_est[i] >= 0:
                y_est[i] = 1
            else:
                y_est[i] = 0
        return y_est

    def fit(self, X, Y, epochs=50):
        _, p = X.shape
        for _ in range(epochs):
            for i in range(p):
                y_est = self.predict(X[:, i].reshape(-1, 1))
                self.w += self.eta * (Y[i] - y_est) * X[:, i]
                self.b += self.eta * (Y[i] - y_est)


def draw_2d_percep(model):
    w1, w2, b = model.w[0], model.w[1], model.b
    plt.plot([-2, 2], [(1 / w2) * (-w1 * (-2) - b), (1 / w2) * (-w1 * 2 - b)], '--k')
    plt.show()


def dataset(num_samples):
    X = np.zeros((2, num_samples))
    Y = np.zeros(num_samples)
    for i in range(num_samples):
        h = 1.2 + (2.2 - 1.2) * np.random.rand()
        w = 50 + (200 - 50) * np.random.rand()
        bmi = w / (h ** 2)
        if bmi > 25:
            ytemp = 1
        else:
            ytemp = 0
        X[:, i] = [h, w]
        Y[i] = ytemp
    return X, Y


# Instanciar el modelo
model = Perceptron(2, 0.1)

# Datos
X, Y = dataset(100)

for i in range(0, 2):
    maxX = max(X[i, :])
    minX = min(X[i, :])
    X[i, :] = (X[i, :] - minX) / (maxX - minX)

# Primero dibujemos los puntos
_, p = X.shape
for i in range(p):
    if Y[i] == 0:
        plt.plot(X[0, i], X[1, i], 'or')
    else:
        plt.plot(X[0, i], X[1, i], 'ob')

# Entrenar
model.fit(X, Y, 100)

# Predicción
model.predict(X)

plt.title('Perceptrón')
plt.grid('on')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')

draw_2d_percep(model)
