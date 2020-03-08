import torch
from torch import nn

# Available device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def squash(x, dim=-1):
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * x / (squared_norm.sqrt() + 1e-8)


class PrimaryCaps(nn.Module):
    """Primary capsule layer."""

    def __init__(self, num_conv_units, in_channels, out_channels, kernel_size, stride):
        super(PrimaryCaps, self).__init__()

        # Each conv unit stands for a single capsule.
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels * num_conv_units,
                              kernel_size=kernel_size,
                              stride=stride)
        self.out_channels = out_channels

    def forward(self, x):
        # Shape of x: (batch_size, in_channels, height, weight)
        # Shape of out: num_capsules * (batch_size, out_channels, height, weight)
        out = self.conv(x)
        # Flatten out: (batch_size, num_capsules * height * weight, out_channels)
        batch_size = out.shape[0]
        return squash(out.contiguous().view(batch_size, -1, self.out_channels), dim=-1)


class DigitCaps(nn.Module):
    """Dynamic routing layer."""

    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, num_iterations):
        super(DigitCaps, self).__init__()
        self.num_route_nodes = num_route_nodes
        self.num_capsules = num_capsules
        self.out_channels = out_channels
        self.route_weights = nn.Parameter(
            torch.randn(num_route_nodes, num_capsules, in_channels, out_channels),
            requires_grad=True)
        self.num_iterations = num_iterations

    def forward(self, x):
        batch_size = x.shape[0]
        # x: (batch_size, num_route_nodes, in_channels)
        # route_weights: (num_route_nodes, num_capsules, in_channels, out_channels)
        # u_hat: (batch_size, num_capsules, num_route_nodes, out_channels)
        u_hat = torch.einsum('ijk, jlkm -> iljm', x, self.route_weights)
        # Detatch u_hat during routing iterations
        u_hat_temp = u_hat.detach()

        # Dynamic route
        # b: (batch_size, num_capsules, num_route_nodes)
        b = torch.zeros(batch_size, self.num_capsules, self.num_route_nodes).to(device)
        for it in range(self.num_iterations - 1):
            c = b.softmax(dim=1)

            # c: (batch_size, num_capsules, num_route_nodes)
            # u_hat: (batch_size, num_capsules, num_route_nodes, out_channels)
            # s: (batch_size, num_capsules, out_channels)
            s = torch.einsum('ijk, ijkl -> ijl', c, u_hat_temp)
            v = squash(s)

            # Update b
            # u_hat: (batch_size, num_capsules, num_route_nodes, out_channels)
            # v: (batch_size, num_capsules, out_channels)
            # Shape of b: (batch_size, num_capsules, num_route_nodes)
            uv = torch.einsum('ijkl, ijl -> ijk', u_hat_temp, v)
            b += uv

        # Last iteration with original u_hat to pass gradient
        c = b.softmax(dim=1)
        s = torch.einsum('ijk, ijkl -> ijl', c, u_hat_temp)
        v = squash(s)

        return v


class CapsNet(nn.Module):
    """Basic implementation of capsule network layer."""

    def __init__(self):
        super(CapsNet, self).__init__()

        # Conv2d layer
        self.conv = nn.Conv2d(1, 256, 9)
        self.relu = nn.ReLU()

        # Primary capsule
        self.primary_caps = PrimaryCaps(num_conv_units=32,
                                        in_channels=256,
                                        out_channels=8,
                                        kernel_size=9,
                                        stride=2)

        # Digit capsule
        self.digit_caps = DigitCaps(num_capsules=10,
                                    num_route_nodes=32 * 6 * 6,
                                    in_channels=8,
                                    out_channels=16,
                                    num_iterations=3)

        # Reconstruction layer
        self.decoder = nn.Sequential(
            nn.Linear(16 * 10, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Sigmoid())

    def forward(self, x, labels):
        out = self.relu(self.conv(x))
        out = self.primary_caps(out)
        out = self.digit_caps(out)

        # Shape of logits: (batch_size, num_capsules)
        logits = torch.norm(out, dim=-1)

        # Reconstruction
        batch_size = out.shape[0]
        reconstruction = self.decoder((out * labels.unsqueeze(2)).contiguous().view(batch_size, -1))

        return logits, reconstruction


class CapsuleLoss(nn.Module):
    """Combine margin loss & reconstruction loss of capsule network."""

    def __init__(self, upper_bound=0.9, lower_bound=0.1, lmda=0.5):
        super(CapsuleLoss, self).__init__()
        self.upper = upper_bound
        self.lower = lower_bound
        self.lmda = lmda
        self.mse = nn.MSELoss()

    def forward(self, images, labels, logits, reconstructions):
        # Shape of left / right / labels: (batch_size, num_classes)
        batch_size = len(labels)
        left = (self.upper - logits).relu() ** 2  # True negative
        right = (logits - self.lower).relu() ** 2  # False positive
        margin_loss = torch.sum(labels * left) + self.lmda * torch.sum((1 - labels) * right)

        # MSE loss for reconstruction
        reconstruction_loss = self.mse(reconstructions, images.view(batch_size, -1))

        # Combine two losses
        return (margin_loss + 0.0005 * reconstruction_loss) / batch_size
