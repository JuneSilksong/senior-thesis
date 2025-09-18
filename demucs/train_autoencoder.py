import numpy as np
np.float_ = np.float64
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import time

from demucs import Demucs
from musdb18 import MUSDB18_Denoising, compare_sources

def train():
    train_dataset = MUSDB18_Denoising()
    print(f"Dataset size: {len(train_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

    time_start = None
    time_finish = None

    model = Demucs(sources=["mix"], resample=False)
    model.cuda()
    model.train()

    batches = int(len(train_dataset)/train_loader.batch_size)

    epochs = 50
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        batch_i = 0
        for noisy, clean in train_loader:
            noisy = noisy.cuda()
            clean = clean.cuda()

            est_clean = model(noisy)
            torch.cuda.synchronize()
            time_forward = time.time()

            loss = F.l1_loss(est_clean, clean)

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

    torch.save(model.state_dict(), 'dae_checkpoint.pth')

"""
def train():
    train_dataset = MUSDB18_Denoising()
    print(f"Dataset size: {len(train_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

    # Model
    model = Demucs(sources=["mix"]).cuda()
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    epochs = 50
    batches = int(len(train_dataset) / train_loader.batch_size)

    time_start = None
    time_finish = None

    # Training
    for epoch in range(epochs):
        batch_i = 0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(cuda), clean.to(device)

            # [B, 2, T] → ensure channel-first format
            if noisy.dim() == 3 and noisy.shape[1] != 2:
                noisy = noisy.permute(0, 2, 1)  
                clean = clean.permute(0, 2, 1)

            est_clean = model(noisy)

            torch.cuda.synchronize()
            time_forward = time.time()

            loss = F.l1_loss(est_clean, clean)

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

    torch.save(model.state_dict(), 'dae_checkpoint.pth')
"""
if __name__ == "__main__":
    train()