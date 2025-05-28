import numpy as np
np.float_ = np.float64
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

from demucs import Demucs
from musdb18 import MUSDB18, compare_sources

def train():
    train_dataset = MUSDB18()
    #train_dataset = Subset(train_dataset, [0])
    print(f"Dataset size: {len(train_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    #train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    model = Demucs(sources=["vocals", "drums", "bass", "other"])
    model.cuda()
    model.train()

    batches = int(len(train_dataset)/train_loader.batch_size)

    epochs = 5
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        batch_i = 0
        for mix, sources in train_loader:
            mix = mix.cuda()
            sources = sources.cuda()

            est_sources = model(mix)

            loss = F.l1_loss(est_sources, sources)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_i += 1
            print(f"Epoch {epoch+1}/{epochs}, Batch {batch_i}/{batches}, Loss = {loss.item():.4f}")
        
        compare_sources(sources, est_sources)
        print(f"Completed Epoch {epoch+1}/{epochs} with Loss = {loss.item():.4f}")

    torch.save(model.state_dict(), 'demucs_checkpoint.pth')

if __name__ == "__main__":
    train()