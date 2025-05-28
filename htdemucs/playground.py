import torch
import matplotlib.pyplot as plt
from demucs import Demucs  # adjust import if needed

# Create random stereo waveform: [B, C, T]
B, C, T = 1, 2, 160000  # 1-second stereo at 16kHz
waveform = torch.randn(B, C, T)

# Initialize model
model = Demucs(sources=["Guitar", "Vocals", "Drums"])

# Run forward pass
with torch.no_grad():
    output, y = model(waveform)

# Plot input and output for first channel
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.title("input")
plt.plot(y[0, 0].cpu().numpy())
plt.subplot(1, 2, 2)
plt.title("output")
plt.plot(output[0, 0].cpu(). numpy())
plt.tight_layout()
plt.show()