from blocks.bricks import Linear, NDimensionalSoftmax
from blocks.bricks.recurrent import LSTM
from blocks.bricks.lookup import LookupTable
from blocks.initialization import NdarrayInitialization, Uniform, Constant
from utils import initialize
from math import sqrt
import theano
import theano.tensor as T
import numpy as np


class NeuralLM:

    def __init__(self, x, y, vocab_size, hidden_size, num_layers, pretrained_embeds=None):
        """
        Implements a neural language model using an LSTM.
        Word y_n+1 ~ Softmax(U * h_n)
        :param x A minibatch: each row is an instance (a sequence),
            with batch_size rows
        :param y x shifted by 1, which are the target words to predict
            for the language modeling objective based on the hidden LSTM
            state
        :param vocab_size The number of types in the training data
        :param hidden_size The dimensionality of the word embeddings
        :param pretrained_embeds Pretrained embeddings for initailization as an ND array
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Initialize the word embedding table.  If we have pretrained embeddings, we use those
        self.word_embedding_lookup = LookupTable(length=vocab_size, dim=hidden_size, name="word_embeddings")
        if pretrained_embeds is None:
            initialize(self.word_embedding_lookup, 0.8)
        else:
            assert pretrained_embeds.shape[0] == vocab_size and pretrained_embeds.shape[1] == hidden_size
            self.word_embedding_lookup.weights_init = Constant(pretrained_embeds)
            self.word_embedding_lookup.biases_init = Constant(0)
            self.word_embedding_lookup.initialize()

        self.word_embeddings = self.word_embedding_lookup.W

        self.y_hat, self.cost, self.cells = self.nn_fprop(x, y, num_layers)

    def lstm_layer(self, h, n):
        """
        Performs the LSTM update for a batch of word sequences
        :param h The word embeddings for this update
        :param n The number of layers of the LSTM
        """
        # Maps the word embedding to a dimensionality to be used in the LSTM
        linear = Linear(input_dim=self.hidden_size, output_dim=self.hidden_size * 4, name='linear_lstm' + str(n))
        initialize(linear, sqrt(6.0 / (5 * self.hidden_size)))
        lstm = LSTM(dim=self.hidden_size, name='lstm' + str(n))
        initialize(lstm, 0.08)
        return lstm.apply(linear.apply(h))

    def softmax_layer(self, h, y):
        """
        Perform Softmax over the hidden state in order to
        predict the next word in the sequence and compute
        the loss.
        :param h The hidden state sequence
        :param y The target words
        """
        hidden_to_output = Linear(name='hidden_to_output', input_dim=self.hidden_size,
                                  output_dim=self.vocab_size)
        initialize(hidden_to_output, sqrt(6.0 / (self.hidden_size + self.vocab_size)))

        linear_output = hidden_to_output.apply(h)
        linear_output.name = 'linear_output'
        softmax = NDimensionalSoftmax(name="lm_softmax")
        y_hat = softmax.log_probabilities(linear_output, extra_ndim=1)
        y_hat.name = 'y_hat'

        cost = softmax.categorical_cross_entropy(y, linear_output, extra_ndim=1).mean()

        cost.name = 'cost'
        return y_hat, cost

    def nn_fprop(self, x, y, num_layers):
        h = T.nnet.sigmoid(self.word_embedding_lookup.apply(x)) # constrain the word embeddings
        cells = []
        for i in range(num_layers):
            h, c = self.lstm_layer(h, i)
            cells.append(c)
        return self.softmax_layer(h, y) + (cells, )

    @property
    def cost(self):
        return self.cost

    @property
    def embeddings(self):
        return self.word_embeddings
