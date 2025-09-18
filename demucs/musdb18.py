import numpy as np
np.float_ = np.float64
import musdb
import torch
from torch.utils.data import Dataset
import librosa
import matplotlib.pyplot as plt
import random
import scipy.signal as spsig

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
        
        sources = torch.stack(sources, dim=0)

        shuffled_sources = np.copy(sources)
        np.random.shuffle(shuffled_sources)
        mix = np.sum(shuffled_sources, axis=0)

        sources = torch.tensor(sources.transpose(0, 2, 1), dtype=torch.float32)
        mix = torch.tensor(mix.T, dtype=torch.float32)
        
        return mix, sources

    def apply_shuffle(self, audio):
        pass

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

class MUSDB18_Denoising(Dataset):
    def __init__(self,
                 root=DATA_PATH,
                 split="train",
                 is_wav=False,
                 segment=10.0,
                 stride=1,
                 mode="mix",  # "mix" or "stems"
                 noise_types=("gaussian", "pink", "crackle", "reverb"),
                 noise_prob=0.7,
                 snr_range=(5, 20)):
        super().__init__()
        self.mus = musdb.DB(root=root, split=split, subsets=split, is_wav=is_wav)
        self.segment = segment
        self.stride = stride
        self.sample_rate = 44100
        self.noise_types = noise_types
        self.noise_prob = noise_prob
        self.snr_range = snr_range
        self.mode = mode
        self.source_names = ["vocals", "drums", "bass", "other"]

        # Precompute segment indices
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

        if self.mode == "mix":
            clean_audio = track.audio[start_sample:end_sample]
            clean = torch.tensor(clean_audio.T, dtype=torch.float32)

            if random.random() < self.noise_prob:
                noisy_audio = self.add_noise(clean_audio)
            else:
                noisy_audio = clean_audio
            noisy = torch.tensor(noisy_audio.T, dtype=torch.float32)

            return noisy, clean

        elif self.mode == "stems":
            clean_stems, noisy_stems = [], []
            for name in self.source_names:
                src_audio = track.targets[name].audio[start_sample:end_sample]
                if random.random() < self.noise_prob:
                    noisy_audio = self.add_noise(src_audio)
                else:
                    noisy_audio = src_audio
                clean_stems.append(torch.tensor(src_audio.T, dtype=torch.float32))
                noisy_stems.append(torch.tensor(noisy_audio.T, dtype=torch.float32))

            clean_stems = torch.stack(clean_stems, dim=0)  # (4, 2, T)
            noisy_stems = torch.stack(noisy_stems, dim=0)
            return noisy_stems, clean_stems

    def add_noise(self, audio):
        """Apply random noise/reverb at a target SNR."""
        noise_type = random.choice(self.noise_types)
        snr_db = random.uniform(*self.snr_range)

        if noise_type == "gaussian":
            noise = np.random.randn(*audio.shape)
        elif noise_type == "pink":
            noise = self.pink_noise(len(audio), audio.shape[1])
        elif noise_type == "crackle":
            noise = (np.random.rand(*audio.shape) < 0.005) * np.random.uniform(-1, 1, audio.shape)
        elif noise_type == "reverb":
            return self.apply_reverb(audio)  # reverb modifies signal instead of additive noise
        else:
            noise = np.zeros_like(audio)

        # scale noise to desired SNR
        clean_power = np.mean(audio**2)
        noise_power = np.mean(noise**2) + 1e-10
        snr_linear = 10**(snr_db / 10)
        scale = np.sqrt(clean_power / (snr_linear * noise_power))
        noisy = audio + scale * noise
        return np.clip(noisy, -1.0, 1.0)

    def apply_reverb(self, audio):
        """Convolve with synthetic impulse response (random decay)."""
        ir_len = random.randint(2000, 8000)  # samples
        ir = np.random.randn(ir_len) * np.exp(-np.linspace(0, 3, ir_len))  # decaying noise
        ir /= np.max(np.abs(ir) + 1e-6)

        # convolve each channel
        rev = []
        for ch in range(audio.shape[1]):
            rev_ch = spsig.fftconvolve(audio[:, ch], ir, mode="full")[:len(audio)]
            rev.append(rev_ch)
        rev = np.stack(rev, axis=1)
        return np.clip(rev, -1.0, 1.0)

    def pink_noise(self, n_samples, n_channels=2):
        uneven = n_samples % 2
        X = (np.random.randn(n_channels, n_samples // 2 + 1 + uneven) +
             1j * np.random.randn(n_channels, n_samples // 2 + 1 + uneven))
        S = np.sqrt(np.arange(len(X[0])) + 1.)  # 1/f spectrum
        y = np.fft.irfft(X / S, n_samples)
        return y.T

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
