import numpy as np
np.float_ = np.float64
import musdb
import torch
from torch.utils.data import Dataset
import librosa
import matplotlib.pyplot as plt

DATA_PATH = 'D:/GitHub/senior-thesis/musdb18'

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

class MUSDB18_Extended(Dataset):
    def __init__(self,
                 root=DATA_PATH,
                 split="train",
                 is_wav=False,
                 segment=11.0,
                 stride=1
                 ):
        super().__init__()
        self.mus = musdb.DB(root=root, split=split, subsets=split, is_wav=is_wav)
        self.segment = segment
        self.stride = stride
        self.sample_rate = 44100

        self.segment_indices = []
        for track_idx, track in enumerate(self.mus):
            max_start = track.duration - self.segment
            if max_start <= 0:
                continue
            num_segments = int(max_start // self.stride) + 1
            for i in range(num_segments):
                start = self.stride * i
                self.segment_indices.append((track_idx, start))
    
    def __len__(self):
        return len(self.segment_indices)
    
    def __getitem__(self, idx):
        track_idx, start = self.segment_indices[idx]
        track = self.mus[track_idx]

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
    
def display_sources(track):
    fig, axs = plt.subplots(5, 1, figsize=(12,10), sharex=True)
    colors = ["gray", "red", "blue", "green", "orange", "gray"]
    i = 1
    for source_name, source in track.targets.items():
        audio = source.audio
        mono_audio = np.mean(audio, axis=1)
        if source_name in ["linear_mixture", "accompaniment"]:
            continue
        axs[i].plot(mono_audio, label=source, color=colors[i], linewidth=0.1)
        axs[i].set_title(source_name)
        axs[i].set_ylabel("Amplitude")
        axs[i].set_xlim(0, len(mono_audio))
        axs[i].set_ylim(-1, 1)
        if source_name in ["vocals", "drums", "bass", "other"]:
            axs[0].plot(mono_audio, label=source, color=colors[i], alpha=0.5, linewidth=0.1)
        i += 1

    axs[0].set_title("waveform")
    axs[-1].set_xlabel("Samples")
    plt.tight_layout()
    plt.show()

def compare_sources(sources, est_sources, mode: str = "waveform"):
    fig, axs = plt.subplots(4, 2, figsize=(10,8), sharex=True, sharey=True)
    colors = ["red", "blue", "green", "orange"]
    source_names = ["vocals","drums","bass","other"]
    if mode not in ["waveform", "spectrogram"]:
        print("Invalid mode")
        return -1
    if mode == "waveform":
        for i in range(len(sources[0])):
            for j, data in enumerate([sources, est_sources]):
                axs[i,j].plot(data[0,i,0].detach().cpu(), color=colors[i])
                axs[i,j].set_title(source_names[i])
                axs[i,j].set_ylabel("Amplitude")
    else:
        for i in range(len(sources[0])):
            for j, data in enumerate([sources, est_sources]):
                y = data[0,i,0]
                y = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
                y = y.astype(np.float32)
                D = librosa.stft(y, n_fft=1024, hop_length=512)
                S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                img = librosa.display.specshow(S_db, ax=axs[i,j])
                fig.colorbar(img, ax=axs[i,j])
                axs[i,j].set_title(source_names[i])
                axs[i,j].set_ylabel("Frequency (Hz)")
    plt.tight_layout()
    plt.show()
