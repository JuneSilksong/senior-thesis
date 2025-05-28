import numpy as np
np.float_ = np.float64
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from demucs import Demucs
from musdb18 import MUSDB18, compare_sources

def train():
    train_dataset = MUSDB18()
    print(f"Dataset size: {len(train_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    model = Demucs(sources=["vocals", "drums", "bass", "other"])
    model.cuda()
    model.train()

    epochs = 2
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        n_track = 0
        for mix, sources in train_loader:
            mix = mix.cuda() # (B, 2, T)
            sources = sources.cuda() # (B, 4, 2, T)

            B, S, C, T = sources.shape

            est_sources = model(mix) # (B, 8, T)
            est_sources = est_sources.view(B, S, C, T) # (B, 4, 2, T)

            loss = F.mse_loss(est_sources, sources)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n_track += 1
            print(f"Epoch {epoch+1}/{epochs}, Track {n_track}, Loss = {loss.item():.4f}")
        
        print(f"Completed Epoch {epoch+1}/{epochs} with Loss = {loss.item():.4f}")
        compare_sources(sources, est_sources)

    torch.save(model.state_dict(), 'demucs_checkpoint.pth')

if __name__ == "__main__":
    train()