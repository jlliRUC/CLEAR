import torch
from torch import nn
import copy
import numpy as np
import pickle


def clones(module, N):
    """
    Produce N identical layers.
    :param module:
    :param N:
    :return:
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def mask_padding(lengths, max_len=None):
    """
    return padding mask based on lengths
    Note that only allow padding in the tail
    :param lengths: (batch_size,)
    :return: mask, bool tensor, (batch_size, max_num_seq), padding position is True
    """
    if max_len is None:
        max_len = int(lengths.max())
    mask = torch.ones((lengths.size()[0], max_len)).cuda()
    for i, len in enumerate(lengths):
        mask[i, len:] = 0
    return mask == 0


def get_parameter_number(model):
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print({'Total': total_num, 'Trainable': trainable_num})


def load_pretrained(filepath, vocab_size):
    if filepath.endswith("txt"):
        with open(filepath) as f:
            # jump the first line, which is the description of the embeddings.
            n_words, dim = [int(value) for value in f.readline().strip().split(" ")]
            embeddings = np.random.randn(vocab_size, dim)
            for line in f:
                word_vec = [float(value) for value in line.strip().split(" ")]
                embeddings[int(word_vec[0])] = word_vec[1:]
        embeddings = torch.tensor(embeddings, dtype=torch.float)
    elif filepath.endswith("pkl"):
        with open(filepath, "rb") as f:
            embeddings = pickle.load(f)

    return embeddings


class MLP(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(MLP, self).__init__()
        self.w_1 = nn.Linear(d_model, d_model * 2)
        self.w_2 = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


