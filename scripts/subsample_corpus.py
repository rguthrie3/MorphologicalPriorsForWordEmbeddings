import argparse
from random import randint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, dest="infile", help="The input to subsample")
    parser.add_argument("-o", required=True, dest="outfile", help="Where to write the subsampled output")
    parser.add_argument("--seq-len", required=True, dest="seq_len", help="Length of sequences to subsample")
    parser.add_argument("--token-count", required=True, dest="token_count", help="Total number of tokens that should be in the output")
    parser.add_argument("--dont-split-on-token", required=False, dest="dont_split_on_token", help="Don't output a sequence that crosses a certain token (e.g ==END-OF-DOCUMENT==")
    options = parser.parse_args()

    data = []
    with open(options.infile, "r") as infile:
        for line in infile:
            data += line.split()

    token_count = int(options.token_count)
    seq_len = int(options.seq_len)
    num_samples = token_count / seq_len
    sample_count = 0
    start_idxs = set([])

    with open(options.outfile, "w") as outfile:
        while sample_count < num_samples:
            start_idx = randint(0, len(data) - seq_len)
            if start_idx in start_idxs:
                continue
            else:
                start_idxs.add(start_idx)
            sample = data[start_idx:start_idx+seq_len]
            if options.dont_split_on_token is not None:
                no_write = False
                for word in sample:
                    if word == options.dont_split_on_token:
                        no_write = True
                        break
            if no_write:
                continue
            outfile.write(' '.join(sample) + "\n")
            sample_count += 1
