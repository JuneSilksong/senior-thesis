import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from stft_wrapper import stft, istft

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

class CrossDomainTransformerEncoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, heads=8):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, t, z):
        B, C, T = z.shape
        z_flat = z.permute(2, 0, 1)  # [T, B, C]
        t_flat = t.permute(2, 0, 1)
        combined = torch.cat([z_flat, t_flat], dim=0)
        encoded = self.transformer(combined)
        # Split and reshape back
        z_out = encoded[:T].permute(1, 2, 0)
        t_out = encoded[T:].permute(1, 2, 0)
        return z_out, t_out

class Demucs(nn.Module):
    def __init__(self):
        super().__init__()
        self.t_enc = nn.ModuleList([
            enc_block(2,48),
            enc_block(48,96),
            enc_block(96,192),
            enc_block(192,384)
        ])
        self.z_enc = nn.ModuleList([
            enc_block(2*2,48),
            enc_block(48,96),
            enc_block(96,192),
            enc_block(192,384)
        ])
        self.t_dec = nn.ModuleList([
            dec_block(384,192),
            dec_block(192,96),
            dec_block(96,48),
            dec_block(48,4*2)
        ])
        self.z_dec = nn.ModuleList([
            dec_block(384,192),
            dec_block(192,96),
            dec_block(96,48),
            dec_block(48,4*2*2)
        ])

        self.stft = stft
        self.istft = istft

        self.CDTE = CrossDomainTransformerEncoder()
    
    def forward(self, x):
        # Time-domain path
        t = x
        t_skips = []
        for encoder in self.t_enc:
            t = encoder(t)
            t_skips.append(t)
        
        # Frequency-domain path
        spec = self.stft(x)                      # [1, 2, 257, 126]
        spec = torch.view_as_real(spec)              # [1, 2, 257, 126, 2]
        spec = spec.permute(0, 1, 4, 3, 2)      # [1, 2, 2, 126, 257]
        spec = spec.reshape(x.size(0), 4, spec.shape[-2], spec.shape[-1])  # [1, 4, 126, 257]
        spec = spec.mean(-1)                    # [1, 4, 126]

        z = spec
        z_skips = []
        for encoder in self.z_enc:
            z = encoder(z)
            z_skips.append(z)
        
        # Cross-Domain Transformer Encoder
        z, t = self.CDTE(z,t)

        # Decoding
        for i, layer in enumerate(self.t_dec):
            t = layer(t + t_skips[-(i+1)])
        for i, layer in enumerate(self.z_dec):
            z = layer(z + z_skips[-(i+1)])

        # Reshape spec and apply ISTFT
        B, C, T = z.shape
        spec_out = z.view(B, 2, -1, T)
        wav_out_from_spec = self.istft(spec_out)
        
        # Combine both outputs (simple average for now)
        output = (wav_out_from_spec + t) / 2
        return output

