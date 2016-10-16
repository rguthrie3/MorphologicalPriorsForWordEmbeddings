# Preprocesses datasets by removing everything except the most common words
# and replacing them with an UNK token in the output
import argparse
from collections import Counter
import re

def preprocess(x):
    if re.match("[0-9]*\.?[0-9][0-9,]+", x) is not None:
        return "NUM"
    else:
        return x.lower()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs=1, help="Text file to preprocess")
    parser.add_argument("-o", dest="output", required=True, help="The output file")
    parser.add_argument("-n", dest="n", default=10000, type=int,
            help="The number of words to preserve in the file (default: 10000)")
    options = parser.parse_args()

    word_counts = Counter()
    with open(options.input[0], "r") as input_file:
        for line in input_file:
            line_tokens = line.split()
            word_counts.update(line_tokens)

    n_most_common_words = set(sorted(word_counts.keys(), key=lambda x: word_counts[x])[-options.n:])
    with open(options.output, "w") as output_file:
        with open(options.input[0], "r") as input_file:
            line_length = 0
            for line in input_file:
                line_tokens = line.split()
                for word in line_tokens:
                    word = preprocess(word)
                    if word in n_most_common_words or word == "NUM":
                        output_file.write(" " + word)
                    else:
                        output_file.write(" UNK")

                    if line_length > 30: # 30 words per line at most
                        line_length = 0
                        output_file.write("\n")
                    else:
                        line_length += 1
