import nltk.tokenize
import argparse
import codecs

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, dest="infile", help="Input text file")
parser.add_argument("-o", required=True, dest="outfile", help="Where to write the outfile")
options = parser.parse_args()

with codecs.open(options.infile, "r", 'utf-8') as infile:
    with codecs.open(options.outfile, "w", 'utf-8') as outfile:
        for line in infile:
            tokenized_line = nltk.tokenize.word_tokenize(line)
            outfile.write(' '.join(tokenized_line) + '\n')
