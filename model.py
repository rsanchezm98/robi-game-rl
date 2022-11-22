import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, num_actions, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)  # apply 32 filters of size 3 so output is o
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=1)
        self.fc3 = nn.Linear(64*5*5, 512)
        self.fc4 = nn.Linear(512, num_actions)
        self.flattener = nn.Flatten()

    def forward(self, x):
        """Build a network that maps state -> action values."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        new_tensor = self.flattener(x)
        x = F.relu(self.fc3(new_tensor))
        output = self.fc4(x)
        return output
