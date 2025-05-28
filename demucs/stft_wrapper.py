import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def stft(x, n_fft=512, hop_length=None, pad=0):
    """
    torch.stft wrapper to perform stft with 3D tensors
    Takes a 3D tensor x of shape (batch_size (B): int=1, channels (C): int=2, time_steps (T): int) e.g. a batch of stereo audio
    and returns the short-time fourier transform z of shape (batch_size (B): int=1, channels (C): int=2, freqs (F): int, frames (N): int)
    i.e. stft : x -> z , T |-> (F,N)
    """
    *other, T = x.shape
    x = x.reshape(-1, T)
    z = torch.stft(x,
                   n_fft * (1 + pad),
                   hop_length or n_fft // 4,
                   window = torch.hann_window(n_fft).to(x),
                   win_length=n_fft,
                   return_complex=True,
                   pad_mode='reflect')
    _, freqs, frames = z.shape
    return z.view(*other, freqs, frames)

def istft(z, hop_length=None, length=None, pad=0):
    """
    torch.istft wrapper to perform istft with 3D tensors
    Takes a 3D tensor z of shape (batch_size (B): int=1, channels (C): int=2, freqs (F): int, frames (N): int)
    and returns the inverse short-time fourier transform x of shape (batch_size (B): int=1, channels (C): int=2, time_steps (T): int)
    i.e. istft : z -> x , (F,N) |-> T
    """
    *other, freqs, frames = z.shape
    n_fft = 2 * freqs - 2
    z = z.view(-1, freqs, frames)
    win_length = n_fft // (1 + pad)
    x = torch.istft(z,
                    n_fft,
                    hop_length,
                    window = torch.hann_window(win_length).to(z.real.device).float(),
                    win_length=win_length,
                    normalized=True,
                    length=length,
                    center=True)
    _, length = x.shape
    return x.view(*other, length)