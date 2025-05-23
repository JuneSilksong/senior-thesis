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

DATA_PATH = 'D:/GitHub/senior-thesis/musdb18/'

model = Demucs(sources=["drums", "bass", "other", "vocals"])
model.load_state_dict(torch.load("demucs_checkpoint.pth", map_location="cuda" if torch.cuda.is_available() else "cpu"))
model.eval()
model.to("cuda" if torch.cuda.is_available() else "cpu")

mus = musdb.DB(root=DATA_PATH, is_wav=False)
track = mus[0]

print(track.name)