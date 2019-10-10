import math


class Ngram:
    def __init__(self, size, *, smoothing = 0, include_terminator = False):
        self.size = size
        self.smoothing = smoothing
        self.include_terminator = include_terminator

        self.vocab = set()
        self.frequencies = dict()

    def add(self, tokens):
        """
        Adds a sequence of tokens to this ngram.

        :param tokens: the token sequence to add
        """

        for token in tokens:
            self.vocab.add(token)

        for leader, token in generate_ngrams(tokens, self.size, include_terminator = self.size != 1):
            if leader not in self.frequencies:
                self.frequencies[leader] = Frequency()

            self.frequencies[leader].add(token)

    def log_probability(self, tokens):
        """
        Determines the probablity of a sequence of tokens in log space.

        :param tokens: the token sequence to get the probablity of

        :return: the probablity of the sequence in log space
        """

        log_sum = 0
        for leader, token in generate_ngrams(tokens, self.size, include_terminator = self.include_terminator):
            if not leader in self.frequencies:
                return float("-inf")

            word_frequency = self.frequencies[leader][token]
            leader_frequency = self.frequencies[leader].total

            probability = (word_frequency + self.smoothing) / (leader_frequency + len(self.vocab) * self.smoothing)

            if probability == 0:
                return float("-inf")

            log_sum += math.log2(probability)
        
        return log_sum

    def probability(self, tokens):
        """
        Determines the probability of a sequence of tokens.

        :param tokens: the token sequence to get the probability of

        :return: the probability of the sequence
        """

        return 2 ** self.log_probability(tokens)


class Frequency:
    """
    Keeps track of the frequncy of a set of items.
    """

    def __init__(self):
        self.frequencies = dict()
        self.total = 0

    def add(self, item):
        if item in self.frequencies:
            self.frequencies[item] += 1
        else:
            self.frequencies[item] = 1

        self.total += 1

    def __getitem__(self, item):
        if item not in self.frequencies:
            return 0

        return self.frequencies[item]


def generate_ngrams(tokens, ngram_size, *, include_terminator = False):
    """
    Generatees ngrams as pairs of leaders and tokens. If include_terminator is
    true an additional token and leader will be returned for the end of the
    sentence with the last token as None.

    :param tokens: the tokens to use to generate ngrams
    :param ngram_size: the size of the ngrams to generate
    :param include_terminator: whether or not to include a None terminator

    :return: ngrams of the given tokens
    """
    
    if ngram_size <= 0:
        raise Exception("ngram size must be positive")

    if not include_terminator:
        for token_index in range(len(tokens)):
            leader = get_leader(tokens, token_index, ngram_size - 1)
            token = tokens[token_index]
            
            yield leader, token
    else:
        for token_index in range(len(tokens) + 1):
            leader = get_leader(tokens, token_index, ngram_size - 1)
            token = tokens[token_index] if token_index < len(tokens) else None

            yield leader, token

def get_leader(tokens, token_index, size):
    """
    Create a tuple out of the tokens that precede the token at the given index.

    If the leader would extend beyond the begging of tokens None will be used.

    :param tokens: the tokens to create the leader from
    :param token_indiex: the index of the token ot get the leader of
    :param size: the number of tokens to include in the leader

    :return: the leader to the token at the given index
    """

    leader = []
    for offset in range(size):
        leader_index = (token_index - 1) - offset
        leader.append(tokens[leader_index] if leader_index >= 0 else None)

    return tuple(leader)
