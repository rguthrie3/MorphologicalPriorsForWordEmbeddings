import argparse
import sys
import cPickle
import numpy as np
from blocks.extensions import saveload
from morfessor import MorfessorIO

def read_data(pickle_file, dict_file):
    with open(pickle_file, "r") as model_f:
        print "Loading Model ", pickle_file, "..."
        trained_model = saveload.load(model_f)
        print "Done"
        param_values = trained_model.model.get_parameter_values()
        word_embeddings = param_values["/word_embeddings.W"]
        morpho_embeddings = param_values["/morpho_embeddings.W"]
    with open(dict_file, "r") as dict_f:
        D = cPickle.load(dict_f)
        word_to_num = D["word_to_ix"]
        morpho_to_num = D["morpho_to_ix"]
    return word_embeddings, morpho_embeddings, word_to_num, morpho_to_num

if __name__ == "__main__":
    sys.path.append("..")

    parser = argparse.ArgumentParser()
    parser.add_argument("-m",
            required=True,
            dest="model_pickle",
            help="Pickle file for the trained Blocks model")
    parser.add_argument("-d",
            required=True,
            dest="dict_pickle",
            help="The dictionaries mapping morphemes and words to integers")
    parser.add_argument("-o",
            required=True,
            dest="output_filename",
            help="The output filename")

    options = parser.parse_args()

    word_embeddings, morpho_embeddings, word_to_ix, morpho_to_ix = read_data(options.model_pickle, options.dict_pickle)
    D = {}
    D["word_to_ix"] = word_to_ix
    D["morpho_to_ix"] = morpho_to_ix
    D["word_embeddings"] = word_embeddings
    D["morpheme_embeddings"] = morpho_embeddings

    with open(options.output_filename, "w") as f:
        cPickle.dump(D, f)
