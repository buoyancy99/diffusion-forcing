import math
import torch.nn as nn
from torch.nn import functional as F


def is_square_of_two(num):
    if num <= 0:
        return False
    return num & (num - 1) == 0

class CnnEncoder(nn.Module):
    """
    Simple cnn encoder that encodes a 64x64 image to embeddings
    """
    def __init__(self, embedding_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.fc = nn.Linear(1024, self.embedding_size)
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        self.modules = [self.conv1, self.conv2, self.conv3, self.conv4]

    def forward(self, observation):
        batch_size = observation.shape[0]
        hidden = self.act_fn(self.conv1(observation))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        hidden = self.act_fn(self.conv4(hidden))
        hidden = self.fc(hidden.view(batch_size, 1024))
        return hidden


class CnnDecoder(nn.Module):
    """
    Simple Cnn decoder that decodes an embedding to 64x64 images
    """
    def __init__(self, embedding_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.fc = nn.Linear(embedding_size, 128)
        self.conv1 = nn.ConvTranspose2d(128, 128, 5, stride=2)
        self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)
        self.modules = [self.conv1, self.conv2, self.conv3, self.conv4]

    def forward(self, embedding):
        batch_size = embedding.shape[0]
        hidden = self.fc(embedding)
        hidden = hidden.view(batch_size, 128, 1, 1)
        hidden = self.act_fn(self.conv1(hidden))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        observation = self.conv4(hidden)
        return observation


class FullyConvEncoder(nn.Module):
    """
    Simple fully convolutional encoder, with 2D input and 2D output
    """
    def __init__(self,
                 input_shape=(3, 64, 64),
                 embedding_shape=(8, 16, 16),
                 activation_function='relu',
                 init_channels=16,
                 ):
        super().__init__()

        assert len(input_shape) == 3, "input_shape must be a tuple of length 3"
        assert len(embedding_shape) == 3, "embedding_shape must be a tuple of length 3"
        assert input_shape[1] == input_shape[2] and is_square_of_two(input_shape[1]), "input_shape must be square"
        assert embedding_shape[1] == embedding_shape[2], "embedding_shape must be square"
        assert input_shape[1] % embedding_shape[1] == 0, "input_shape must be divisible by embedding_shape"
        assert is_square_of_two(init_channels), "init_channels must be a square of 2"

        depth = int(math.sqrt(input_shape[1] / embedding_shape[1])) + 1
        channels_per_layer = [init_channels * (2 ** i) for i in range(depth)]
        self.act_fn = getattr(F, activation_function)

        self.downs = nn.ModuleList([])
        self.downs.append(nn.Conv2d(input_shape[0], channels_per_layer[0], kernel_size=3, stride=1, padding=1))

        for i in range(1, depth):
            self.downs.append(nn.Conv2d(channels_per_layer[i-1], channels_per_layer[i],
                                        kernel_size=3, stride=2, padding=1))

        # Bottleneck layer
        self.downs.append(nn.Conv2d(channels_per_layer[-1], embedding_shape[0], kernel_size=1, stride=1, padding=0))

    def forward(self, observation):
        hidden = observation
        for layer in self.downs:
            hidden = self.act_fn(layer(hidden))
        return hidden


class FullyConvDecoder(nn.Module):
    """
    Simple fully convolutional decoder, with 2D input and 2D output
    """
    def __init__(self,
                 embedding_shape=(8, 16, 16),
                 output_shape=(3, 64, 64),
                 activation_function='relu',
                 init_channels=16,
                 ):
        super().__init__()

        assert len(embedding_shape) == 3, "embedding_shape must be a tuple of length 3"
        assert len(output_shape) == 3, "output_shape must be a tuple of length 3"
        assert output_shape[1] == output_shape[2] and is_square_of_two(output_shape[1]), "output_shape must be square"
        assert embedding_shape[1] == embedding_shape[2], "input_shape must be square"
        assert output_shape[1] % embedding_shape[1] == 0, "output_shape must be divisible by input_shape"
        assert is_square_of_two(init_channels), "init_channels must be a square of 2"

        depth = int(math.sqrt(output_shape[1] / embedding_shape[1])) + 1
        channels_per_layer = [init_channels * (2 ** i) for i in range(depth)]
        self.act_fn = getattr(F, activation_function)

        self.ups = nn.ModuleList([])
        self.ups.append(nn.ConvTranspose2d(embedding_shape[0], channels_per_layer[-1],
                                           kernel_size=1, stride=1, padding=0))

        for i in range(1, depth):
            self.ups.append(nn.ConvTranspose2d(channels_per_layer[-i], channels_per_layer[-i-1],
                                               kernel_size=3, stride=2, padding=1, output_padding=1))

        self.output_layer = nn.ConvTranspose2d(channels_per_layer[0], output_shape[0],
                                               kernel_size=3, stride=1, padding=1)

    def forward(self, embedding):
        hidden = embedding
        for layer in self.ups:
            hidden = self.act_fn(layer(hidden))

        return self.output_layer(hidden)
