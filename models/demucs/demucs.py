import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import typing as tp
import math

def center_trim(tensor: torch.Tensor, reference: tp.Union[torch.Tensor, int]):
    """
    Center trim `tensor` with respect to `reference`, along the last dimension.
    `reference` can also be a number, representing the length to trim to.
    If the size difference != 0 mod 2, the extra sample is removed on the right side.
    """
    ref_size: int
    if isinstance(reference, torch.Tensor):
        ref_size = reference.size(-1)
    else:
        ref_size = reference
    delta = tensor.size(-1) - ref_size
    if delta < 0:
        raise ValueError("tensor must be larger than reference. " f"Delta is {delta}.")
    if delta:
        tensor = tensor[..., delta // 2:-(delta - delta // 2)]
    return tensor


def enc_block(c_in, c_out, kernel=4, stride=2, norm_groups=4):
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, kernel_size=kernel, stride=stride, padding=kernel//2),
        nn.GroupNorm(norm_groups, c_out),
        nn.GELU()
    )

def dec_block(c_in, c_out, kernel=4, stride=2, norm_groups=4):
    return nn.Sequential(
        nn.ConvTranspose1d(c_in, c_out, kernel_size=kernel, stride=stride, padding=kernel//2),
        nn.GroupNorm(norm_groups, c_out),
        nn.GELU()
    )

class Demucs(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.ModuleList([
            enc_block(2,48),
            enc_block(48,96),
            enc_block(96,192),
            enc_block(192,384)
        ])
        self.dec = nn.ModuleList([
            dec_block(384,192),
            dec_block(192,96),
            dec_block(96,48),
            dec_block(48,4*2*2)
        ])
        self.resample = True
        self.depth = 4
        self.kernel_size = 8
        self.stride = 4
        self.sources = ["one","two"]
        self.audio_channels = 2

    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolution, e.g. for all
        layers, size of the input - kernel_size % stride = 0.

        Note that input are automatically padded if necessary to ensure that the output
        has the same length as the input.
        """
        if self.resample:
            length *= 2

        for _ in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(1, length)

        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size

        if self.resample:
            length = math.ceil(length / 2)
        return int(length)
    
    def forward(self, x):
        # Time-domain path
        length = x.shape[-1]
        
        mono = x.mean(dim=1, keepdim=True)
        mean = mono.mean(dim=-1, keepdim=True)
        std = mono.std(dim=-1, keepdim=True)
        x = (x - mean) / (1e-5 + std)

        delta = self.valid_length(length) - length
        x = F.pad(x, (delta // 2, delta - delta // 2))

        saved = []
        for encoder in self.enc:
            x = encoder(x)
            saved.append(x)

        # Decoding
        for decoder in self.dec:
            skip = saved.pop(-1)
            skip = center_trim(skip,x)
            x = decoder(x + skip)

        x = x * std + mean
        x = center_trim(x, length)
        x = x.view(x.size(0), len(self.sources), self.audio_channels, x.size(-1))
        return x
