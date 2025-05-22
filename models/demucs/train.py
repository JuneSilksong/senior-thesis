import numpy as np
np.float_ = np.float64
import musdb
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim
from demucs import Demucs

DATA_PATH = 'D:/GitHub/senior-thesis/musdb18'

mus = musdb.DB(root=DATA_PATH, is_wav=False) # 150 full length stereo music tracks encoded at 44.1 kHz
track = mus[0]

def display_sources(track):
    fig, axs = plt.subplots(6, 1, figsize=(12,10), sharex=True)
    colors = ["gray", "red", "blue", "green", "orange", "gray"]
    i = 1
    for source_name, source in track.targets.items():
        audio = source.audio
        mono_audio = np.mean(audio, axis=1)
        if source_name == "accompaniment":
            continue
        axs[i].plot(mono_audio, label=source, color=colors[i], linewidth=0.1)
        axs[i].set_title(source_name)
        axs[i].set_ylabel("Amplitude")
        axs[i].set_xlim(0, len(mono_audio))
        axs[i].set_ylim(-1, 1)
        if source_name in ["vocals", "drums", "bass", "other"]:
            axs[0].plot(mono_audio, label=source, color=colors[i], alpha=0.5, linewidth=0.1)
        i += 1

    axs[-1].set_xlabel("Samples")
    plt.tight_layout()
    plt.show()

class MUSDB18(Dataset):
    def __init__(self,
                 root=DATA_PATH,
                 split="train",
                 is_wav=False,
                 segment=10.0
                 ):
        super().__init__()
        self.mus = musdb.DB(root=root, split=split, subsets=split, is_wav=is_wav)
        self.segment = segment
        self.sample_rate = 44100
    
    def __len__(self):
        return len(self.mus)
    
    def __getitem__(self, idx):
        track = self.mus[idx]
        start = np.random.randint(0, track.duration - self.segment)
        start_sample = int(start * self.sample_rate)
        end_sample = start_sample + int(self.segment * self.sample_rate)

        mix_audio = track.audio[start_sample:end_sample]
        mix = torch.tensor(mix_audio.T, dtype=torch.float32)

        sources = []
        for source_name in ["vocals", "drums", "bass", "other"]:
            source_audio = track.targets[source_name].audio[start_sample:end_sample]
            source = torch.tensor(source_audio.T, dtype=torch.float32)
            sources.append(source)
        
        sources = torch.stack(sources, dim=0)
        
        return mix, sources

train_dataset = MUSDB18()
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True)

model = Demucs(sources=["vocals", "drums", "bass", "other"])
model.cuda()
epochs = 10
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(epochs):
    for mix, sources in train_loader:
        mix = mix.cuda()              # (B, C, T)
        sources = sources.cuda()      # (B, 4, C, T)

        B, S, C, T = sources.shape
        sources = sources.view(B, S * C, T)  # (B, 8, T)

        est_sources = model(mix)            # (B, 8, T)

        loss = F.l1_loss(est_sources, sources)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), 'demucs_checkpoint.pth')