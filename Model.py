import numpy as np
import tqdm

from CyclicLR import CyclicLR

class MnistClassificator():
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights and biases for the first linear layer
        self.weights_1 = np.random.randn(self.input_size, self.hidden_size)
        self.biases_1 = np.zeros((1, self.hidden_size))

        # Initialize weights and biases for the second linear layer
        self.weights_2 = np.random.randn(self.hidden_size, self.output_size)
        self.biases_2 = np.zeros((1, self.output_size))

    def linear(self, x, weights, biases):
        return np.dot(x, weights) + biases

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def mse_loss(self, y_true, y_pred):
        return ((y_true - y_pred)**2).mean()

    def forward(self, X):
        # Forward pass
        hidden_layer_output = self.relu(self.linear(X, self.weights_1, self.biases_1))
        output_layer_output = self.softmax(self.linear(hidden_layer_output, self.weights_2, self.biases_2))

        return output_layer_output, hidden_layer_output

    def backward(self, X_batch, y_true_batch, output_layer_output, hidden_layer_output):
        m = X_batch.shape[0]

        # Backpropagation through the second linear layer
        output_error = output_layer_output - y_true_batch
        grad_weights_2 = np.dot(hidden_layer_output.T, output_error)
        grad_biases_2 = np.sum(output_error, axis=0, keepdims=True)

        # Backpropagation through the first linear layer with ReLU activation
        hidden_layer_error = np.dot(output_error, self.weights_2.T) * (hidden_layer_output > 0)
        grad_weights_1 = np.dot(X_batch.T, hidden_layer_error)
        grad_biases_1 = np.sum(hidden_layer_error, axis=0, keepdims=True)

        # Update weights and biases using gradient descent
        self.weights_1 -= self.learning_rate * grad_weights_1 / m
        self.biases_1 -= self.learning_rate * grad_biases_1 / m
        self.weights_2 -= self.learning_rate * grad_weights_2 / m
        self.biases_2 -= self.learning_rate * grad_biases_2 / m

    def train(self, X_train, y_train, epochs=10, batch_size=32, cyclic_lr=True, min_lr=0.001, max_lr=0.01):
        m = X_train.shape[0]

        if cyclic_lr:
            clr = CyclicLR(min_lr=min_lr, max_lr=0.01, step_size=8 * (m // batch_size))

        for epoch in range(epochs):
            if cyclic_lr:
                self.learning_rate = clr.update()  # Update learning rate if using cyclic LR

            # Shuffle the data for each epoch
            permutation = np.random.permutation(m)
            X_train_shuffled = X_train.iloc[permutation]
            y_train_shuffled = y_train[permutation]

            # Mini-batch training
            for i in tqdm.trange(0, m, batch_size):
                X_batch = X_train_shuffled.iloc[i:i+batch_size].values
                y_batch = y_train_shuffled[i:i+batch_size]

                output_layer_output, hidden_layer_output = self.forward(X_batch)
                self.backward(X_batch, y_batch, output_layer_output, hidden_layer_output)

            # Calculate and print the loss after each epoch
            output_layer_output, _ = self.forward(X_train.values)
            loss = self.mse_loss(y_train, output_layer_output)
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Learning Rate: {self.learning_rate:.6f}')


    def predict(self, X):
        # Forward pass for prediction
        output_layer_output, _ = self.forward(X)
        return output_layer_output