import torch
import matplotlib.pyplot as plt
import numpy as np

from stft_wrapper import stft, istft

sample_rate = 16000
duration = 1
t = torch.linspace(0, duration, sample_rate * duration, dtype=torch.float32)

# Construct waveform
hz = [1000, 2000, 4000, 6000]
amp = [0.8, 0.1, 0.3, 0.6]
waveform = 0

for i in range(len(hz)):
    waveform += amp[i] * torch.sin(2*np.pi*hz[i]*t)
waveform = waveform.unsqueeze(0).unsqueeze(0)

# Perform stft and istft
spec = stft(waveform)
x = istft(spec)

print(waveform.shape)
print(spec.shape)
print(x.shape)

# Create spectrogram
spec_db = 20 * torch.log10(spec.abs() + 1e-6)

# Plot spectrogram
plt.figure(figsize=(12, 12))

plt.subplot(3, 1, 1)
plt.plot(t.numpy(), waveform[0, 0].numpy())
plt.title("Time-Domain Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 2)
plt.imshow(spec_db[0,0].cpu(), origin='lower', aspect='auto', cmap='magma', extent=[0, duration, 0, sample_rate // 2])
plt.colorbar(label="Magnitude (dB)")
plt.title("STFT Spectrogram")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")

plt.subplot(3, 1, 3)
plt.plot(t.numpy(), x[0, 0].numpy())
plt.title("Reconstructed Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")


plt.tight_layout()
plt.show()