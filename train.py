import theano
import numpy as np
import cPickle
import argparse
import sys
from theano import tensor
from blocks.model import Model
from blocks.graph import ComputationGraph, apply_dropout
from blocks.algorithms import RMSProp, StepClipping, GradientDescent, CompositeRule
from blocks.filter import VariableFilter
from blocks.extensions import FinishAfter, Timing, Printing, saveload
from blocks.extensions.training import SharedVariableModifier
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.monitoring import aggregation
from utils import get_metadata, get_stream, track_best, MainLoop, read_pretrained_vectors, DecayIfIncrease
from model.neural_lm_model import NeuralLM
from model.morpho_model import MorphoPrior
from config import config
import logging

logging.basicConfig(level=logging.DEBUG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs=1, help="The training hdf5 file")
    parser.add_argument("-pre", dest="pre", help="Pretrained word embeddings file in word2vec format (word <space> embedding, one per line)")
    options = parser.parse_args()

    # Load config parameters
    locals().update(config)
    logging.debug('loaded config')

    # DATA
    hdf5_file = options.input[0]
    word_to_ix, ix_to_word, morpho_to_ix, ix_to_morpho = get_metadata(hdf5_file)
    vocab_size = len(word_to_ix)
    morpho_vocab_size = len(morpho_to_ix)
    train_stream = get_stream(hdf5_file, 'train', batch_size)
    dev_stream = get_stream(hdf5_file, 'dev', batch_size)
    logging.debug('loaded data')
    print "Number of words:", vocab_size
    print "Number of morphemes:", morpho_vocab_size
    # Save the word and morpheme indices to disk
    D = { }
    D["word_to_ix"] = word_to_ix
    D["morpho_to_ix"] = morpho_to_ix
    cPickle.dump(D, open("dicts.pkl", "w"))
    logging.debug('wrote dicts')
    # Load the pretrained vectors if available
    if options.pre is not None:
        pretrained_ndarray = read_pretrained_vectors(options.pre, word_to_ix)
    else:
        pretrained_ndarray = None

    # MODEL
    broadcastable_4D_tensor = tensor.TensorType("uint8", (False, False, False, True)) # Need 4D broadcast for masking
    word_features = tensor.matrix('word_features', dtype='uint32')
    morpho_features = tensor.tensor3('morpho_features', dtype='uint32')
    morpho_masks = broadcastable_4D_tensor('morpho_masks')
    word_targets = tensor.matrix('word_targets', dtype='uint32')
    neural_lm = NeuralLM(word_features, word_targets, vocab_size, hidden_size, num_layers, pretrained_ndarray)
    morpho_prior = MorphoPrior(morpho_features, morpho_masks, word_features, morpho_vocab_size, hidden_size, neural_lm.embeddings)

    # COST
    cost = neural_lm.cost + morpho_prior.cost
    cost.name = "cost"
    cg = ComputationGraph(cost)
    logging.debug('made computation graph')

    if dropout > 0:
        # Apply dropout only to the non-recurrent inputs (Zaremba et al. 2015)
        inputs = VariableFilter(theano_name_regex=r'.*apply_input.*')(cg.variables)
        cg = apply_dropout(cg, inputs, dropout)
        cost = cg.outputs[0]

    # Learning algorithm
    step_rules = [RMSProp(learning_rate=learning_rate, decay_rate=decay_rate),
                  StepClipping(step_clipping)]
    algorithm = GradientDescent(cost=cost, parameters=cg.parameters,
                                step_rule=CompositeRule(step_rules), on_unused_sources="ignore")
    logging.debug('made learning algo')

    # Extensions (stuff to track during training, serializing the training loop to disk, etc.)
    monitored_vars = [cost, morpho_prior.morpheme_vector_norm, morpho_prior.cost]
    dev_monitor = DataStreamMonitoring(variables=[cost], after_epoch=True,
                                       before_first_epoch=True, data_stream=dev_stream, prefix="dev")
    train_monitor = TrainingDataMonitoring(variables=monitored_vars, after_batch=True,
                                           before_first_epoch=True, prefix='tra')
    #decay_if_increase = DecayIfIncrease(step_rules[0].learning_rate, "dev_cost", after_epoch=True)

    extensions = [dev_monitor, train_monitor, Timing(), Printing(after_batch=True),
                  FinishAfter(after_n_epochs=nepochs),
                  saveload.Load(load_path),
                  saveload.Checkpoint(last_path),
                  #decay_if_increase,
                  ] + track_best('dev_cost', save_path)
    logging.debug('made extensions')

    if learning_rate_decay not in (0, 1): # Note: This is the tuple (0,1), not the interval.
        extensions.append(SharedVariableModifier(step_rules[0].learning_rate,
            lambda n, lr: np.cast[theano.config.floatX](learning_rate_decay * lr), after_epoch=True, after_batch=False))

    logging.info('number of parameters in the model: ' + str(tensor.sum([p.size for p in cg.parameters]).eval()))

    # Finally build the main loop and train the model
    main_loop = MainLoop(data_stream=train_stream, algorithm=algorithm,
                         model=Model(cost), extensions=extensions)
    main_loop.run()
