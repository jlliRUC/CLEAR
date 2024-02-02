# This consists of a naive RNN module and projection head
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence
from utils_model import mask_padding, MLP


# Encoder
class DualRNNEncoder(nn.Module):
    def __init__(self, attn, embed_s, embed_c, hidden_size, bidirectional, dropout, num_layers):
        super(DualRNNEncoder, self).__init__()
        self.embed_s = embed_s
        self.embed_c = embed_c
        self.attn = attn
        self.num_directions = 2 if bidirectional else 1
        assert hidden_size % self.num_directions == 0
        self.hidden_size = hidden_size // self.num_directions
        self.dropout = dropout
        if num_layers == 1:
            self.dropout = None
        self.rnn_s = nn.GRU(self.embed_s.embed_size,
                            self.hidden_size,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            batch_first=True)
        self.rnn_c = nn.GRU(self.embed_s.embed_size,
                            self.hidden_size,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            batch_first=True)

        self.mlp = MLP(self.hidden_size * self.num_directions, dropout)

    def single_encode(self, x, lengths, rnn, h0):
        # RNN Encoding
        if self.attn is None:
            # embed -> RNN
            # return hn of RNN, hn will be adjusted to the last layer of hn from RNN
            # RNN
            if lengths is not None:
                x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            # hn: (num_layers * num_directions, batch_size, hidden_size)
            output, hn = rnn(x, h0)
            if lengths is not None:
                output = pad_packed_sequence(output, batch_first=True)[0]

            # hn: (batch_size, num_directions*hidden_size)
            hn = self.adjust(hn)
            return hn
        else:
            # embed -> RNN -> attn
            # return weighted sum of output of RNN
            # RNN
            if lengths is not None:
                x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            # output: (batch_size, num_seq, num_directions*hidden_size)
            output, hn = rnn(x, h0)
            if lengths is not None:
                output = pad_packed_sequence(output, batch_first=True)[0]
            # output: (batch_size, num_seq, num_directions*hidden_size)
            # score: (batch_size, num_seq, num_seq) (average weights across heads)
            output, score = self.attn(query=output, key=output, value=output, key_padding_mask=mask_padding(lengths))

            # output: (batch_size, num_directions*hidden_size)
            output = torch.sum(output, dim=1)
            return output

    def forward(self, locations, lengths, h0_s=None, h0_c=None):
        # embed
        # (batch_size, num_seq) -> (batch_size, num_seq, embed_size)
        if self.embed_c is not None:  # dual RNN
            # s
            x_s = self.embed_s(locations)
            x_s = self.single_encode(x_s, lengths, self.rnn_s, h0_s)

            # c
            x_c = self.embed_c(locations)
            x_c = self.single_encode(x_c, lengths, self.rnn_c, h0_c)

            return x_s + self.mlp(x_c)
        else:
            x = self.embed_s(locations)
            return self.single_encode(x, lengths, self.rnn_s, h0_s)

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


if __name__ == "__main__":
    num_head = 4
    embed_size = 256
    hidden_size = 256
    num_layers = 1
    dropout = 0.1
    vocab_size = 18888
    bidirectional = True

    from utils_model import get_parameter_number
    from Embedding import SEmbed
    from ProjectionHead import ProjectionHead
    from CLEAR import CLEAR

    embed_s = SEmbed(vocab_size, embed_size)
    embed_c = None
    encoder = DualRNNEncoder(None, embed_s, embed_c, hidden_size, bidirectional, dropout, num_layers)
    projector = ProjectionHead(embed_size, int(embed_size / 2), int(embed_size / 4), batch_norm=True)

    model = CLEAR(encoder, projector)

    get_parameter_number(model)
