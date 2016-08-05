import sys
import h5py
import yaml
from fuel.datasets import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme
from fuel.transformers import Mapping
from blocks.extensions import saveload, predicates
from blocks.extensions.training import TrackTheBest
from blocks import initialization
from blocks import main_loop
from fuel.utils import do_not_pickle_attributes
import numpy as np
from config import config

# Load config parameters
locals().update(config)

#Define this class to skip serialization of extensions
@do_not_pickle_attributes('extensions')
class MainLoop(main_loop.MainLoop):

    def __init__(self, **kwargs):
        super(MainLoop, self).__init__(**kwargs)

    def load(self):
        self.extensions = []


def transpose_stream(data):
    ret = [0]*4
    ret[0] = np.transpose(data[0], axes=[1,0,2])
    ret[1] = np.transpose(data[1], axes=[1,0,2]).reshape(data[1].shape[1], data[1].shape[0], max_morphemes_per_word, 1)
    ret[2] = data[2].T
    ret[3] = data[3].T
    return tuple(ret)


def track_best(channel, save_path):
    tracker = TrackTheBest(channel, choose_best=min)
    checkpoint = saveload.Checkpoint(
        save_path, after_training=False, use_cpickle=True)
    checkpoint.add_condition(["after_epoch"],
                             predicate=predicates.OnLogRecord('{0}_best_so_far'.format(channel)))
    return [tracker, checkpoint]


def get_metadata(hdf5_file):
    with h5py.File(hdf5_file) as f:
        word_to_ix = yaml.load(f['word_targets'].attrs['word_to_ix'])
        ix_to_word = yaml.load(f['word_targets'].attrs['ix_to_word'])

        morpho_to_ix = yaml.load(f['word_targets'].attrs['morpho_to_ix'])
        ix_to_morpho = yaml.load(f['word_targets'].attrs['ix_to_morpho'])
    return word_to_ix, ix_to_word, morpho_to_ix, ix_to_morpho


def get_stream(hdf5_file, which_set, batch_size=None):
    dataset = H5PYDataset(
        hdf5_file, which_sets=(which_set,), load_in_memory=True)
    if batch_size == None:
        batch_size = dataset.num_examples
    stream = DataStream(dataset=dataset, iteration_scheme=ShuffledScheme(
        examples=dataset.num_examples, batch_size=batch_size))
    # Required because Recurrent bricks receive as input [sequence, batch,
    # features]
    return Mapping(stream, transpose_stream)


def read_pretrained_vectors(filename, word_to_ix):
    with open(filename, "r") as f:
        lines = f.readlines()
    word_to_embed = { line.split()[0]: map(lambda x: float(x), line.split()[1:]) for line in lines }
    sorted_word_indices = sorted(word_to_ix.items(), key=lambda x: x[1])
    output_arr = [ word_to_embed[w] for w, i in sorted_word_indices ]
    return np.array(output_arr)


def initialize(to_init, width):
    """
    Initialize weights according to Xavier Parameter Initialization
    :param to_init the block to initialize
    :param width width of uniform distribution
    """
    to_init.weights_init = initialization.Uniform(width=width)
    to_init.biases_init = initialization.Constant(0)
    to_init.initialize()
