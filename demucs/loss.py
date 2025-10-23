import torch
import torch.nn.functional as F

def si_sdr_loss(estimate, target, eps=1e-8):
    """
    Channel-wise SI-SDR loss for stereo (or multi-channel) audio.
    Supports input shape (B, C, T)
    """
    B, C, T = estimate.shape

    # Zero-mean per sample
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    # Scale and projection
    target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + eps
    scale = torch.sum(estimate * target, dim=-1, keepdim=True) / target_energy
    projection = scale * target

    noise = estimate - projection
    ratio = torch.sum(projection ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + eps)
    si_sdr = 10 * torch.log10(ratio + eps)  # shape: (B, C)

    return -si_sdr.mean()

def stft(x, n_fft, hop_length, win_length, window):
    B, C, T = x.shape
    x = x.view(B * C, T)
    
    x_stft = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)
    
    x_stft = x_stft.reshape(B, C, x_stft.shape[-2], x_stft.shape[-1])
    
    return torch.abs(x_stft)

def stft_loss(est, ref, n_fft, hop_length, win_length):
    """
    Perform STFT and convert to magnitude spectrogram.
    """
    window = torch.hann_window(win_length, device=est.device)
    est_stft = stft(est, n_fft, hop_length, win_length, window)
    ref_stft = stft(ref, n_fft, hop_length, win_length, window)

    sc_loss = torch.norm(ref_stft - est_stft, p='fro') / (torch.norm(ref_stft, p='fro') + 1e-8)
    mag_loss = F.l1_loss(torch.log(ref_stft + 1e-8), torch.log(est_stft + 1e-8))

    return sc_loss + mag_loss

def mrstft_loss(est, ref):
    fft_params = [
        (1024, 120, 600),
        (2048, 240, 1200),
        (512, 50, 240),
    ]
    loss = 0.0
    for n_fft, hop, win in fft_params:
        loss += stft_loss(est, ref, n_fft, hop, win)
    return loss / len(fft_params)