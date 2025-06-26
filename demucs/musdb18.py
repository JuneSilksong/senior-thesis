import numpy as np
np.float_ = np.float64
import musdb
import torch
from torch.utils.data import Dataset
import librosa
import matplotlib.pyplot as plt
import random

# DATA_PATH = 'D:/GitHub/senior-thesis/musdb18' # Windows
DATA_PATH = '/home/user/Github/musdb18' # Ubuntu

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

class MUSDB18_ExtAugmented(Dataset):
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
        self.source_names = ["vocals", "drums", "bass", "other"]
        self.augment = True

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

        offset = random.uniform(0, 1.0)
        start += offset

        start_sample = int(start * self.sample_rate)
        end_sample = start_sample + int(self.segment * self.sample_rate)

        sources = []
        for source_name in self.source_names:
            source_audio = track.targets[source_name].audio[start_sample:end_sample]
            if self.augment:
                source_audio = self.apply_random_scale(source_audio)
                source_audio = self.apply_random_flip(source_audio)
                source_audio = self.apply_channel_swap(source_audio)
            sources.append(source_audio)
        
        sources = np.stack(sources, axis=0)
        mix = np.sum(sources, axis=0)

        sources = torch.tensor(sources.transpose(0, 2, 1), dtype=torch.float32)
        mix = torch.tensor(mix.T, dtype=torch.float32)
        
        return mix, sources

    def apply_channel_swap(self, audio):
        if random.random() < 0.5:
            return audio[:, [1, 0]]
        return audio

    def apply_random_scale(self, audio):
        gain = random.uniform(0.25,1.25)
        return audio*gain

    def apply_random_flip(self, audio):
        if random.random() < 0.5:
            return -audio
        return audio
    
def collate_fn(batch):
    batch_size = len(batch)
    num_sources = 4

    all_sources = torch.stack([item[1] for item in batch], dim=0)

    shuffled_sources = []
    for src_idx in range(num_sources):
        src_group = all_sources[:, src_idx]
        indices = torch.randperm(batch_size)
        shuffled_group = src_group[indices]
        shuffled_sources.append(shuffled_group)

    new_sources = torch.stack(shuffled_sources, dim=0).transpose(0, 1)  # shape: [B, 4, 2, T]
    new_mix = torch.sum(new_sources, dim=1)

    return new_mix, new_sources

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
    fig.suptitle("Ground Truth (left) vs Estimate (right)")
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
                y = y.detdach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
                y = y.astype(np.float32)
                D = librosa.stft(y, n_fft=1024, hop_length=512)
                S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                img = librosa.display.specshow(S_db, ax=axs[i,j])
                fig.colorbar(img, ax=axs[i,j])
                axs[i,j].set_title(source_names[i])
                axs[i,j].set_ylabel("Frequency (Hz)")
    plt.tight_layout()
    plt.show()