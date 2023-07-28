import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence
import constants


def mask_padding(lengths):
    """
    return padding mask based on lengths
    Note that only allow padding in the tail
    :param lengths: (batch_size,)
    :return: mask, bool tensor, (batch_size, max_num_seq), padding position is True
    """
    max_len = int(lengths.max())
    mask = torch.ones((lengths.size()[0], max_len)).cuda()
    for i, len in enumerate(lengths):
        mask[i, len:] = 0
    return mask == 0


# Embed
class SEmbed(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(SEmbed, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.embed = nn.Embedding(num_embeddings=self.vocab_size,
                                  embedding_dim=self.embed_size,
                                  padding_idx=constants.PAD)

    def forward(self, locations):
        return self.embed(locations)


# Encoder
class SEncoder(nn.Module):
    def __init__(self, attn, embed_s, hidden_size, bidirectional, dropout, num_layers):
        super(SEncoder, self).__init__()
        self.embed_s = embed_s
        self.attn = attn
        self.num_directions = 2 if bidirectional else 1
        assert hidden_size % self.num_directions == 0
        self.hidden_size = hidden_size // self.num_directions
        self.dropout = dropout
        if num_layers == 1:
            self.dropout = None
        self.rnn = nn.GRU(self.embed_s.embed_size,
                          self.hidden_size,
                          num_layers=num_layers,
                          bidirectional=bidirectional,
                          dropout=dropout,
                          batch_first=True)

    def forward(self, locations, lengths, h0=None):
        # embed -> RNN
        # return hn of RNN, hn will be adjusted to the last layer of hn from RNN
        # embed
        # (batch_size, num_seq) -> (batch_size, num_seq, embed_size)
        x = self.embed_s(locations)

        # RNN
        if lengths is not None:
            x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        # hn: (num_layers * num_directions, batch_size, hidden_size)
        output, hn = self.rnn(x, h0)
        if lengths is not None:
            output = pad_packed_sequence(output, batch_first=True)[0]

        # hn: (batch_size, num_directions*hidden_size)
        hn = self.adjust(hn)
        return output, hn

    def adjust(self, vec):
        """
        we only use the last layer of hn
        :param vec: hn of RNN (num_layers * num_directions, batch_size, hidden_size)
        :return: (batch_size, num_directions*hidden_size)
        """
        if self.num_directions == 2:
            num_layers, batch_size, hidden_size = vec.size(0) // 2, vec.size(1), vec.size(2)
            # (num_layers, batch, hidden_size * num_directions)
            vec = vec.view(num_layers, 2, batch_size, hidden_size) \
                .transpose(1, 2).contiguous() \
                .view(num_layers, batch_size, hidden_size * 2)
        else:
            vec = vec
        return vec[-1]


# Projection Head
class ProjectionHead(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_norm=True, head_type='nonlinear'):
        super().__init__()
        self.lin1 = nn.Linear(input_size, hidden_size, bias=True)
        self.bn = nn.BatchNorm1d(hidden_size) if batch_norm else nn.Identity()
        self.relu = nn.ReLU()
        # No bias for the final linear layer
        self.lin2 = nn.Linear(hidden_size, output_size, bias=False)
        self.head_type = head_type
        self.lins = nn.Sequential(self.lin1, self.bn, self.relu, self.lin2) if self.head_type == 'nonlinear' \
                    else nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        """

        :param x: (batch_size, input_size)
        :return: (batch_size, output_size)
        """
        return self.lins(x)


# Clear
class SClear(nn.Module):
    """
    Contrastive Learning + Spatial information
    """

    def __init__(self, encoder, projector):
        super(SClear, self).__init__()
        self.encoder = encoder
        self.projector = projector

    def forward(self, locations, lengths=None):
        _, h = self.encoder(locations, lengths)
        z = self.projector(h)

        return h, z


