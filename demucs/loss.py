import torch
import torch.nn.functional as F

def si_sdr_loss(est, ref, eps=1e-8):
    """Scale-invariant SDR loss (negative SI-SDR)."""
    ref_energy = torch.sum(ref ** 2, dim=-1, keepdim=True)
    proj = torch.sum(est * ref, dim=-1, keepdim=True) * ref / (ref_energy + eps)
    noise = est - proj
    ratio = torch.sum(proj ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + eps)
    return -10 * torch.log10(ratio + eps).mean()

def mrstft_loss(est, ref):
    """Multi-resolution STFT loss."""
    total = 0
    configs = [(1024, 256), (2048, 512), (512, 128)]
    for n_fft, hop in configs:
        est_stft = torch.stft(est, n_fft=n_fft, hop_length=hop,
                              return_complex=True, center=True)
        ref_stft = torch.stft(ref, n_fft=n_fft, hop_length=hop,
                              return_complex=True, center=True)
        mag_est, mag_ref = est_stft.abs(), ref_stft.abs()
        sc = torch.norm(mag_ref - mag_est, p='fro') / (torch.norm(mag_ref, p='fro') + 1e-8)
        mag = F.l1_loss(torch.log(mag_est + 1e-8), torch.log(mag_ref + 1e-8))
        total += sc + mag
    return total / len(configs)