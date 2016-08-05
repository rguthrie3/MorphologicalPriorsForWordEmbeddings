WORD_VECTORS_VOCAB_FILE = "wordvectorsorg/wordvectorsorg_vocab.txt"

class WordVectorsOrgInterface:


    def __init__(self):
        pass


    @staticmethod
    def output_word_vector(word, vector, f):
        """
        Output a vector in the format needed for wordvectors.org
        to the given text file.
        """
        f.write(word + " ")
        for i in vector:
            f.write(str(i) + " ")
        f.write("\n")


    @staticmethod
    def parse_vocab():
        """
        Returns a list of all the words in
        the total wordvectors.org vocab
        """
        ret = set([])
        with open(WORD_VECTORS_VOCAB_FILE, "r") as f:
            for word in f:
                ret.add(word.strip())
        return list(ret)
