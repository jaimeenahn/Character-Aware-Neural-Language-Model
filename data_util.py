import torch
import numpy as np
from torch.autograd import Variable


def word_to_idx(words, word_vocab):
    idx = []
    for word in words:
        idx += [word_vocab[word]]

    return np.array(idx)


def to_char(x, char_vocab, max_len):
    for i, word in enumerate(x):
        chars = [char_vocab[c] for c in list(word)]
        chars.insert(0, char_vocab['{'])
        chars.append(char_vocab['}'])
        for k in range(0, max_len - len(chars)):
            chars.append(0)

        x[i] = chars


def txt_to_word(path, batch_size, trunc_step):
    data = []

    with open(path, 'r') as f:
        for line in f:
            words = line.split() + ['+']
            data += words

    total_len = len(data)
    num_batches = total_len // (batch_size * trunc_step)
    data = data[:num_batches * batch_size * trunc_step]

    return data, num_batches
