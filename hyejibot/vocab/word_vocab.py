

class WordVocab:
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}
        self.word2cnt = {}
        self.n_words = 0

    def add_word(self, word):
        if word in self.word2idx.keys():
            self.word2cnt[word] += 1
        else:
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.word2cnt[word] = 1
            self.n_words += 1