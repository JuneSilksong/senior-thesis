import numpy as np
np.float_ = np.float64
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import time

from demucs import Demucs
from musdb18 import MUSDB18, MUSDB18_Extended, compare_sources

time_initial = time.time()

def train():
    train_dataset = MUSDB18_Extended()
    #train_dataset = Subset(train_dataset, [0])
    print(f"Dataset size: {len(train_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    #train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    time_start = None
    time_finish = None

    model = Demucs(sources=["vocals", "drums", "bass", "other"], resample=False)
    model.cuda()
    model.train()

    batches = int(len(train_dataset)/train_loader.batch_size)

    epochs = 50
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        batch_i = 0
        for mix, sources in train_loader:
            mix = mix.cuda()
            sources = sources.cuda()

            est_sources = model(mix)
            torch.cuda.synchronize()
            time_forward = time.time()

            loss = F.l1_loss(est_sources, sources)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            time_backward = time.time()

            batch_i += 1
            
            torch.cuda.synchronize()
            time_finish = time.time()
            print(f"Forward split: {(time_forward-time_start if time_start else time_forward):.4f}")
            print(f"Backward split {(time_backward-time_forward):.4f}")
            print(f"Epoch {epoch+1}/{epochs}, Batch {batch_i}/{batches}, Loss = {loss.item():.4f}, Duration = {(time_finish-time_start if time_start else time_finish):.4f}")
            time_start = time.time()
            
        
        print(f"Completed Epoch {epoch+1}/{epochs} with Loss = {loss.item():.4f}")

    torch.save(model.state_dict(), 'demucs_checkpoint.pth')

if __name__ == "__main__":
    train()