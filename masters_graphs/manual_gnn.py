import numpy as np
import torch

# Transfer neuron activation
def sigmoid(input):
    return 1.0 / (1.0 + np.exp(-input))


# Calculate the derivative of an neuron output
def sigmoid_derivative(val):
    return sigmoid(val) * (1.0 - sigmoid(val))


def softmax_grad(softmax):
    s = softmax.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)


features_and_y = np.array(
    [
        [0, 0, 0],
        [0, 0, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]
)
features = features_and_y[:, :-1]
y = features_and_y[:, -1]

assoc = np.eye(6) + np.array(
    [
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0],
    ]
)

relu = lambda x: np.maximum(x, 0)
weights = np.random.rand(2)
weights_neigh = [1, 1]

# learning_rate = 0.01
# for i in range(100):
#     print("weights:", weights)
#     neighbor_features = assoc @ features @ weights_neigh
#     node_features = assoc @ features @ weights
#     # print(neighbor_features)
#     # print("my features:", features.dot(weights))

#     z = node_features + neighbor_features.mean()
#     output = sigmoid(z)
#     error = (output - y) * softmax_grad(z)

#     error_neigh = (output - y) * softmax_grad(neighbor_features)
#     print(error)
#     print("error:", error.sum(0))
#     weights -= (error @ features * learning_rate).sum(0)
#     weights_neigh -= (error_neigh @ features * learning_rate).sum(0)
#     print(1)
# print(weights)
# print(output)
import numpy as np


class NeuralNetwork:
    def __init__(self):
        np.random.seed(10)  # for generating the same results
        self.wij = np.random.rand(3, 1)  # input to hidden layer weights
        self.wjk = np.random.rand(4, 1)  # hidden layer to output weights

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, x, w):
        return self.sigmoid(x @ w) * (1 - self.sigmoid(x @ w))

    def gradient_descent(self, x, y, iterations):
        for i in range(iterations):
            Xi = x
            Xj = self.sigmoid(Xi @ self.wij)
            yhat = self.sigmoid(Xj @ self.wjk)
            # gradients for hidden to output weights
            g_wjk = Xj.T @ (y - yhat) * self.sigmoid_derivative(Xj, self.wjk)
            # gradients for input to hidden weights
            g_wij = Xi.T @ (
                (((y - yhat) * self.sigmoid_derivative(Xj, self.wjk)) @ self.wjk.T)
                * self.sigmoid_derivative(Xi, self.wij)
            )

            # update weights
            self.wij += g_wij
            self.wjk += g_wjk
        print("The final prediction from neural network are: ")
        print(yhat)


if __name__ == "__main__":
    neural_network = NeuralNetwork()
    print("Random starting input to hidden weights: ")
    print(neural_network.wij)
    print("Random starting hidden to output weights: ")
    print(neural_network.wjk)
    X = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    y = np.array([[0, 1, 1, 0]]).T
    neural_network.gradient_descent(X, y, 10000)
