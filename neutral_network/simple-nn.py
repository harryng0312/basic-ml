import numpy as np


class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(y.shape)
        self.layer1 = None

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T,
                            (2 * (self.y - self.output) * sigmoid_derivative(self.output)))
        tmp = (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output),
                      self.weights2.T) * sigmoid_derivative(self.layer1))
        d_weights1 = np.dot(self.input.T, tmp)

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2


def sigmoid(S):
    """
    S: an numpy array
    return sigmoid function of each element of S
    """
    return 1 / (1 + np.exp(-S))


def sigmoid_derivative(S):
    return sigmoid(S) * (1 - sigmoid(S))


def tanh(S):
    """
    S: an numpy array
    return sigmoid function of each element of S
    """
    return (np.exp(S) - np.exp(-S)) / (np.exp(S) + np.exp(-S))


X = np.array([[0, 0, 1, 1], [0, 1, 0, 1], [1, 1, 1, 1]])
Y = np.array([0, 1, 1, 0])
nn = NeuralNetwork(X, Y)
nn.feedforward()
nn.backprop()
print(f"output:\n{nn.weights1}\n")
print(f"W:\n{nn.weights1}\n\n{nn.weights2}")
