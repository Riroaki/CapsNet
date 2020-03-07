import torch
from torch import nn


def squash(x, dim=-1):
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * x / squared_norm.sqrt()


class PrimaryCaps(nn.Module):
    """Primary capsule layer."""

    def __init__(self, num_conv_units, in_channels, out_channels, kernel_size, stride):
        super(PrimaryCaps, self).__init__()

        # Each conv unit stands for a single capsule.
        self.capsules = nn.ModuleList([nn.Conv2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride)
                                       for _ in range(num_conv_units)])

    def forward(self, x):
        # Shape of x: (batch_size, in_channels, height, weight)
        # Shape of out: num_capsules * (batch_size, out_channels, height, weight)
        out = [capsule(x) for capsule in self.capsules]
        # Shape of out: (batch_size, num_capsules, height, weight, out_channels)
        out = torch.stack(out, dim=0).permute(1, 0, 3, 4, 2)
        # Flatten out: (batch_size, num_capsules * height * weight, out_channels)
        return squash(out.flatten(start_dim=1, end_dim=-2), dim=-1)


class DigitCaps(nn.Module):
    """Dynamic routing layer."""

    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, num_iterations):
        super(DigitCaps, self).__init__()
        self.num_capsules = num_capsules
        self.out_channels = out_channels
        self.route_weights = nn.Parameter(torch.rand(num_route_nodes, in_channels, out_channels),
                                          requires_grad=True)
        self.num_iterations = num_iterations

    def forward(self, x):
        # Shape of u_hat: (batch_size, num_capsules * height * weight, out_channels)
        u_hat = torch.einsum('ijk, jkl -> ijl', x, self.route_weights)

        # Dynamic route
        b = torch.zeros(x.shape[1], self.num_capsules, requires_grad=True)
        for it in range(self.num_iterations):
            c = b.softmax(dim=-1)

            # Shape of s / v: (batch_size, num_capsules, out_channels)
            s = torch.einsum('ijk, jl -> ilk', u_hat, c)
            v = squash(s, dim=-1)

            # Update b
            if it < self.num_iterations - 1:
                b = b + torch.einsum('ijk, ilk -> jl', u_hat, v)
            else:
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

    def forward(self, x):
        out = self.relu(self.conv(x))
        out = self.primary_caps(out)
        out = self.digit_caps(out)

        # Shape of logits: (batch_size, num_capsules)
        logits = torch.norm(out, dim=-1)

        # Reconstruction
        reconstruction = self.decoder(out.flatten(start_dim=1))

        return logits, reconstruction


class CapsuleLoss(nn.Module):
    """Margin loss & reconstruction loss of capsule network."""

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
        margin_loss = torch.sum(labels * left)
        margin_loss += torch.sum(self.lmda * (1 - labels) * right)

        # MSE loss for reconstruction
        reconstruction_loss = self.mse(reconstructions, images.flatten(start_dim=1))

        # Combine two losses
        return (margin_loss + 0.0005 * reconstruction_loss) / batch_size
