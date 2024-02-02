import sys
sys.path.append("../")
from torch import nn
import constants


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


