'''
Reservoir sampling 
main() will perform this sampling on stdin
Vitter 1985, "Algorithm R"
Consider implementing Algorithm Z sometime
'''

from random import randint
from collections import Counter, deque
import argparse
import codecs
import nltk.tokenize
import re

class SequenceGenerator:

    def __init__(self, filename, seq_len):
        self.seq_len = seq_len
        self.file = codecs.open(filename, "r", "utf-8", "utf-8")
        self.buffer = deque()
        self.line = self.file.readline()
        self.done = False

    def __iter__(self):
        return self

    def next(self):
        if self.done:
            raise StopIteration
        seq = []
        while len(self.buffer) != 0 and len(seq) < self.seq_len:
            seq.append(self.buffer.popleft())
        if len(seq) == self.seq_len:
            return seq

        while self.line != "" and len(seq) < self.seq_len:
            tokenized = nltk.tokenize.word_tokenize(self.line, language='danish')
            for token in tokenized:
                if len(seq) < self.seq_len:
                    seq.append(token)
                else:
                    self.buffer.append(token)
            self.line = self.file.readline()
        if len(seq) < self.seq_len:
            self.done = True
            raise StopIteration
        else:
            return seq


def res_sample(gen,k):
    output = [None]*k
    for i in xrange(k):
        output[i] = gen.next()
    counter = 0
    for thing in gen:
        r = randint(0,counter+k)
        if r < k:
            output[r] = thing
        counter += 1
    return output


def preprocess(x):
    if re.match("([0-9]*[\.,]?[0-9]+)", x) is not None:
        return "NUM"
    else:
        return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs=1, help="Text file to process")
    parser.add_argument("-o", dest="output", required=True, help="Output filename")
    parser.add_argument("--num-types", dest="num_types", default=50000, help="Number of unique types to preserve if doing preprocessing")
    parser.add_argument("--no-subsample", dest="do_subsample", default=True, action="store_false", help="Whether or not to subsample the text")
    parser.add_argument("--subsample-token-count", required=True, dest="token_count", help="Total number of tokens that should be in the output")
    parser.add_argument("--subsample-seq-len", dest="seq_len", default=50, help="Length of the chunks that will be sampled")
    parser.add_argument("--subsample-dont-split-on-token", required=False, dest="dont_split_on_token", help="Don't output a sequence that crosses a certain token (e.g ==END-OF-DOCUMENT==")
    options = parser.parse_args()

    # Subsample the data
    num_samples = int(options.token_count) / int(options.seq_len)
    seq_generator = SequenceGenerator(options.input[0], int(options.seq_len))

    sampled_data = res_sample(seq_generator, num_samples)

    # Preprocess the data
    word_counts = Counter()
    for sequence in sampled_data:
        word_counts.update(sequence)

    n_most_common_words = set(sorted(word_counts.keys(), key=lambda x: word_counts[x])[-int(options.num_types):])
    with codecs.open(options.output, "w", "utf-8") as output_file:
        for sequence in sampled_data:
            for word in sequence:
                word = preprocess(word)
                if word in n_most_common_words or word == "NUM":
                    output_file.write(" " + word)
                else:
                    output_file.write(" UNK")
            output_file.write("\n")


if __name__ == "__main__":
    main()

