import numpy
import codecs
import h5py
import yaml
import argparse
import os
from fuel.datasets import H5PYDataset
from config import config
from morfessor.io import MorfessorIO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-mf", required=True, help="The morfessor model to use",
            dest="morf_model")
    parser.add_argument("input", nargs=1, help="Input text file")
    options = parser.parse_args()

    # Load config parameters
    locals().update(config)
    numpy.random.seed(0)

    # Read in the morfessor model
    morf_segmenter = MorfessorIO().read_binary_model_file(options.morf_model)

    with codecs.open(options.input[0], 'r', 'utf-8') as f:
        data = f.read().split()

    if len(data) % seq_length > 0:
        data = data[:len(data) - len(data) % seq_length + 1]
    else:
        data = data[:len(data) - seq_length + 1]
    nsamples = len(data) // seq_length

    # Read in word-level data
    words = set(data)
    vocab_size = len(words)
    word_to_ix = {word: i for i, word in enumerate(words)}
    ix_to_word = {i: word for i, word in enumerate(words)}

    # Read in morpho-level data
    morphos = set([])
    cache = {}
    max_morpheme_num = 0
    for word in data:
        if word in cache:
            morpheme_decomp = cache[word]
        else:
            morpheme_decomp = morf_segmenter.viterbi_segment(word)[0]
            cache[word] = morpheme_decomp
            if len(morpheme_decomp) > max_morpheme_num:
                max_morpheme_num = len(morpheme_decomp)
        for m in morpheme_decomp:
            morphos.add(m)
    print "Word with most morphemes has", max_morpheme_num, "morphemes"
    morpho_vocab_size = len(morphos)
    morpho_to_ix = {morpho: i for i, morpho in enumerate(morphos)}
    ix_to_morpho = {i: morpho for i, morpho in enumerate(morphos)}

    word_inputs = numpy.empty((nsamples, seq_length), dtype='uint32')
    morpho_inputs = numpy.empty((nsamples, seq_length, max_morpheme_num), dtype='uint32')
    morpho_masks_inputs = numpy.zeros_like(morpho_inputs)
    word_outputs = numpy.zeros_like(word_inputs)
    for i, p in enumerate(range(0, len(data) - 1, seq_length)):
        word_inputs[i] = numpy.array([word_to_ix[word] for word in data[p:p + seq_length]])
        morpho_inputs[i] = numpy.array([ [morpho_to_ix[m] for m in cache[word]] + [0] * (max_morpheme_num - len(cache[word])) for word in data[p:p + seq_length]])
        morpho_masks_inputs[i] = numpy.array([ [1] * len(cache[word]) + [0] * (max_morpheme_num - len(cache[word])) for word in data[p:p + seq_length]])
        word_outputs[i] = numpy.array([word_to_ix[word] for word in data[p + 1:p + seq_length + 1]])

    f = h5py.File(os.path.splitext(options.input[0])[0] + ".hdf5", mode='w')
    word_features = f.create_dataset('word_features', word_inputs.shape, dtype='uint32')
    morpho_features = f.create_dataset('morpho_features', morpho_inputs.shape, dtype='uint32')
    morpho_masks = f.create_dataset('morpho_masks', morpho_masks_inputs.shape, dtype='uint8')
    word_targets = f.create_dataset('word_targets', word_outputs.shape, dtype='uint32')

    word_targets.attrs['word_to_ix'] = yaml.dump(word_to_ix)
    word_targets.attrs['ix_to_word'] = yaml.dump(ix_to_word)
    word_targets.attrs['morpho_to_ix'] = yaml.dump(morpho_to_ix)
    word_targets.attrs['ix_to_morpho'] = yaml.dump(ix_to_morpho)

    word_features[...] = word_inputs
    morpho_features[...] = morpho_inputs
    morpho_masks[...] = morpho_masks_inputs
    word_targets[...] = word_outputs
    word_features.dims[0].label = 'batch'
    word_features.dims[1].label = 'sequence'
    word_targets.dims[0].label = 'batch'
    word_targets.dims[1].label = 'sequence'

    nsamples_train = int(nsamples * train_size)
    split_dict = {
        'train': {'word_features': (0, nsamples_train),
                  'word_targets': (0, nsamples_train),
                  'morpho_features': (0, nsamples_train),
                  'morpho_masks': (0, nsamples_train)},
        'dev': {'word_features': (nsamples_train, nsamples),
                'word_targets': (nsamples_train, nsamples),
                'morpho_features': (nsamples_train, nsamples),
                'morpho_masks': (nsamples_train, nsamples)}}

    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
    f.flush()
    f.close()
