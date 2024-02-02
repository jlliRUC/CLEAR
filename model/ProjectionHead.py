from torch import nn


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
