import torch
import matplotlib.pyplot as plt
from ht_demucs import Demucs  # adjust import if needed

# Create random stereo waveform: [B, C, T]
B, C, T = 1, 2, 16000  # 1-second stereo at 16kHz
waveform = torch.randn(B, C, T)

# Initialize model
model = Demucs()

# Run forward pass
with torch.no_grad():
    output = model(waveform)

# Plot input and output for first channel
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.title("Input waveform (ch 0)")
plt.plot(waveform[0, 0].cpu().numpy())
plt.subplot(1, 2, 2)
plt.title("Output waveform (ch 0)")
plt.plot(output[0, 0].cpu().numpy())
plt.tight_layout()
plt.show()