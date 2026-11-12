"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.power_scalar(1 + ops.exp(-x), -1)
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size, self.hidden_size = input_size, hidden_size
        self.nonlinearity = nonlinearity
        bound = 1 / np.sqrt(hidden_size)
        self.W_ih = Parameter(init.rand(input_size, hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
        self.bias_ih = Parameter(init.rand(hidden_size, low=-bound, high=bound, device=device, dtype=dtype)) if bias else None
        self.bias_hh = Parameter(init.rand(hidden_size, low=-bound, high=bound, device=device, dtype=dtype)) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            h = init.zeros(X.shape[0], self.hidden_size, device=X.device, dtype=X.dtype)
        y = X @ self.W_ih + h @ self.W_hh
        if self.bias_ih is not None:
            y = y + self.bias_ih.reshape((1, self.hidden_size)).broadcast_to(y.shape)
            y = y + self.bias_hh.reshape((1, self.hidden_size)).broadcast_to(y.shape)
        return ops.tanh(y) if self.nonlinearity == "tanh" else ops.relu(y)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.hidden_size, self.num_layers = hidden_size, num_layers
        self.rnn_cells = [RNNCell(input_size if i == 0 else hidden_size, hidden_size, bias=bias, nonlinearity=nonlinearity, device=device, dtype=dtype) for i in range(num_layers)]
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        states = list(ops.split(h0, 0)) if h0 is not None else [None] * self.num_layers
        outputs = []
        for x in ops.split(X, 0):
            for layer, cell in enumerate(self.rnn_cells):
                states[layer] = cell(x, states[layer])
                x = states[layer]
            outputs.append(x)
        return ops.stack(outputs, 0), ops.stack(states, 0)
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size, self.hidden_size = input_size, hidden_size
        bound = 1 / np.sqrt(hidden_size)
        self.W_ih = Parameter(init.rand(input_size, 4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
        self.W_hh = Parameter(init.rand(hidden_size, 4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
        self.bias_ih = Parameter(init.rand(4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype)) if bias else None
        self.bias_hh = Parameter(init.rand(4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype)) if bias else None
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            h = (init.zeros(X.shape[0], self.hidden_size, device=X.device, dtype=X.dtype), init.zeros(X.shape[0], self.hidden_size, device=X.device, dtype=X.dtype))
        h0, c0 = h
        gates = X @ self.W_ih + h0 @ self.W_hh
        if self.bias_ih is not None:
            gates = gates + self.bias_ih.reshape((1, 4 * self.hidden_size)).broadcast_to(gates.shape)
            gates = gates + self.bias_hh.reshape((1, 4 * self.hidden_size)).broadcast_to(gates.shape)
        gates = ops.split(gates.reshape((X.shape[0], 4, self.hidden_size)), 1)
        i, f, g, o = [gate for gate in gates]
        sigmoid = Sigmoid()
        c = sigmoid(f) * c0 + sigmoid(i) * ops.tanh(g)
        return sigmoid(o) * ops.tanh(c), c
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.hidden_size, self.num_layers = hidden_size, num_layers
        self.lstm_cells = [LSTMCell(input_size if i == 0 else hidden_size, hidden_size, bias=bias, device=device, dtype=dtype) for i in range(num_layers)]
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            states = [None] * self.num_layers
        else:
            states = list(zip(ops.split(h[0], 0), ops.split(h[1], 0)))
        outputs = []
        for x in ops.split(X, 0):
            for layer, cell in enumerate(self.lstm_cells):
                states[layer] = cell(x, states[layer])
                x = states[layer][0]
            outputs.append(x)
        return ops.stack(outputs, 0), (ops.stack([state[0] for state in states], 0), ops.stack([state[1] for state in states], 0))
        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
        self.weight = Parameter(init.randn(num_embeddings, embedding_dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        one_hot = init.one_hot(self.num_embeddings, x, device=self.weight.device, dtype=self.weight.dtype)
        return (one_hot.reshape((-1, self.num_embeddings)) @ self.weight).reshape(x.shape + (self.embedding_dim,))
        ### END YOUR SOLUTION
