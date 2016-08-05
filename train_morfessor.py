"""
Trains a morfessor morphological segmenter
for use in the morphological RNN
"""
from morfessor.io import MorfessorIO
from morfessor.baseline import BaselineModel

import argparse
import subprocess
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data", required=True, dest="training_data", help="Corpus to train Morfessor on")
    parser.add_argument("--output", dest="output", default="./morfessor_model", help="Output filename for the Morfessor model")
    options = parser.parse_args()

    data_reader = MorfessorIO()
    word_iterator = data_reader.read_corpus_file(options.training_data)
    model = BaselineModel()
    model.load_data(word_iterator, count_modifier=lambda x: 1) # Use types instead of tokens
    model.train_batch()

    data_reader.write_binary_model_file(options.output + "-" + os.path.basename(options.training_data) + ".bin", model)
