import torch.utils.data as data
import torch
import pickle
import random
from hyejibot.vocab.phoneme_vocab import PhonemeVocab
from hyejibot.vocab.word_vocab import WordVocab


class DomainDataset(data.Dataset):
    """Domain Custom Dataset"""
    def __init__(self, source, target, src_sent2idx, trg_word2idx):
        self.source = source
        self.target = target
        self.src_sent2idx = src_sent2idx
        self.trg_word2idx = trg_word2idx

    def __getitem__(self, index):
        src = self.src_sent2idx(self.source[index])
        trg = self.trg_word2idx[self.target[index]]
        return torch.tensor(src), torch.tensor(trg)

    def __len__(self):
        return len(self.target)

def collate_fn(data):
    """Creates mini-batch from the list

    Args:
        data: list of tuple (src, trg)
    """

    def merge(sequences, src=None):
        lengths = [len(seq) for seq in sequences]
        if src:
            padded_seqs = torch.zeros(len(sequences), max(lengths), src[0], src[1]).long()
        else:
            padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # seperate source and target sequences
    src_seqs, trg_seqs = zip(*data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(list(src_seqs), src=(3,2))
    trg_seqs = torch.stack(trg_seqs)
    return src_seqs, src_lengths, trg_seqs


def get_ratio_range(num_data, ratio):
    sum_ratio = 0
    ratio_range = []
    for r in ratio:
        sum_ratio += r
        ratio_range.append(int(sum_ratio * num_data))
    return [0] + ratio_range


def get_loaders(file_path, batch_size, ratio):

    with open(file_path, "br") as r:
        data = pickle.load(r)

    src_lang = PhonemeVocab()
    trg_lang = WordVocab()
    src = []
    trg = []
    for k, v in data.items():
        trg_lang.add_word(k)
        src += v
        trg += [k for _ in range(len(v))]

    # shuffle data
    d = list(zip(src, trg))
    random.shuffle(d)
    src, trg = zip(*d)
    src, trg = list(src), list(trg)

    r_range = get_ratio_range(len(src), ratio)
    data_loaders = []
    for r in range(len(ratio)-1):
        sub_src = src[r_range[r]:r_range[r+1]]
        sub_trg = trg[r_range[r]:r_range[r+1]]
        dataset = DomainDataset(sub_src, sub_trg, src_lang.sent2idx, trg_lang.word2idx)
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size[r],
                                                  shuffle=True,
                                                  drop_last=True,
                                                  collate_fn=collate_fn)
        data_loaders.append(data_loader)
    return data_loaders, src_lang, trg_lang

