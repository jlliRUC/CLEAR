from torch import nn


class CLEAR(nn.Module):
    """
    Contrastive Learning + Spatial information
    """

    def __init__(self, encoder, projector):
        super(CLEAR, self).__init__()
        self.encoder = encoder
        self.projector = projector

    def forward(self, locations, lengths=None):
        h = self.encoder(locations, lengths)
        z = self.projector(h)

        return h, z
