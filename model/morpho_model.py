from blocks.bricks.lookup import LookupTable
from blocks.bricks import NDimensionalSoftmax
from utils import initialize
import theano.tensor as T

class MorphoPrior:

    def __init__(self, morpho_idxs, masks, word_idxs, morpho_vocab_size, hidden_size, word_embeds):
        """
        Implements a morpheme-level prior by computing the sum of KL-Div
        of the elements of the morpheme embeddings and the word embeddings
        (where these elements are in [0,1] and are taken as Bernoulli dists).
        :param morpho_idxs A 3D tensor of batch_size x seq_length x max_morphemes_per_word
            Where the 3rd dimension is morpheme indices, padded with 0's so all words have
            the same morpheme decomposition length
        :param masks A 4D tensor of bits which select which values in morpho_idxs are
            padding and which are actual morphemes.  4D is needed for broadcasting
        :param word_idxs A 2D matrix of batch_size x seq_length of word indices
        :param morpho_vocab_size the number of morpheme types seen in training data
        :param hidden_size the dimensionality of morpheme / word embeddings
        :param word_embeds the unconstrained word embeddings from the language model
        """
        self.morpho_vocab_size = morpho_vocab_size
        self.hidden_size = hidden_size
        self.word_embed_lookup = word_embeds # These are the unconstrained word embeddings

        self.morpho_embed_lookup = LookupTable(length=morpho_vocab_size,
                dim=hidden_size, name="morpho_embeddings")
        self.softmax = NDimensionalSoftmax(name="morpho_softmax")
        initialize(self.morpho_embed_lookup, 0.8)

        self.cost = self.compute_cost(morpho_idxs, masks, word_idxs)
        self.cost.name = "morpho_cost"

    def compute_cost(self, morpho_idxs, masks, word_idxs):
        """
        Compute the cost by taking a softmax over the morpheme embedding,
        the word embedding, then computing the KL divergence of them
        :param morpho_idxs A 3D tensor where morpho_idxs[i,j] is a word,
            and the 3rd dimension is the morpheme indicies that make up
            the word.
        :param masks A tensor of masks for the morphemes, where each entry
            is 1 or 0 indicating whether a given morpheme index is padding
            or an actual morpheme index of the word
        :param word_idxs A 2D matrix of word indices
        """
        output_shape = [word_idxs.shape[i] for i in range(word_idxs.ndim)] + [self.hidden_size]
        word_level_reps = self.word_embed_lookup[word_idxs.flatten()].reshape(output_shape)
        morpho_level_reps = (self.morpho_embed_lookup.apply(morpho_idxs) * masks).sum(axis=2)
        return self.kl_div(word_level_reps, morpho_level_reps)

    def kl_div(self, x, y):
        """
        Compute sum of D(x_i || y_i) for each corresponding element
        along the 3rd dimension (the embedding dimension)
        of x and y
        This function takes care to not compute logarithms that are close
        to 0, since NaN's could result for log(sigmoid(x)) if x is negative.
        It simply uses that log(sigmoid(x)) = - log(1 + e^-x)
        """
        sig_x = T.nnet.sigmoid(x)
        exp_x       = T.exp(-x)
        exp_y       = T.exp(-y)
        one_p_exp_x = exp_x + 1
        one_p_exp_y = exp_y + 1
        return (sig_x * (T.log(one_p_exp_y) - T.log(one_p_exp_x)) + (1 - sig_x) * (T.log(exp_y) - T.log(exp_x))).mean()


    @property
    def cost(self):
        return self.cost

    @property
    def embeddings(self):
        return self.morpho_embed_lookup
