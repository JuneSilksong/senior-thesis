import numpy as np
np.float_ = np.float64
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from musdb import DB
from types import SimpleNamespace

from demucs import Demucs
from musdb18 import MUSDB18, display_sources

# DATA_PATH = 'D:/GitHub/senior-thesis/musdb18/'
DATA_PATH = '/home/user/Github/musdb18'
CHECKPOINT_PATH = '/home/user/Github/senior-thesis/pth/demucs_20250613_10.pth'

mus = DB(root=DATA_PATH, is_wav=False)  # test set has no ground truth
print(f"Found {len(mus)} tracks.")
track = mus[0]

display_sources(track)

model = Demucs(sources=["drums", "bass", "other", "vocals"])
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(checkpoint['model_state_dict'])
model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

sample_rate = 44100
audio = track.audio.T
audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).cuda()

with torch.no_grad():
    est_sources = model(audio_tensor)

est_sources = est_sources.view(4,2,-1).cpu().numpy()

source_names = ["vocals","drums","bass","other"]
track_targets = {}

for i,name in enumerate(source_names):
    audio = est_sources[i].T
    track.targets[name] = SimpleNamespace(audio=audio)

display_sources(track)