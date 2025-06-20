import numpy as np
np.float_ = np.float64
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import time

from demucs import Demucs
from musdb18 import MUSDB18, MUSDB18_Extended#, compare_sources

#print("Hello")
print(torch.backends.cudnn.version() if torch.backends.cudnn.version() else "No version")
time_initial = time.time()

print(torch.cuda.is_available())  # Should return True
print(torch.cuda.device_count())  # Should return at least 1

num_workers = 16

def train(model_path: str):
    train_dataset = MUSDB18_Extended(split="train")
    #train_dataset = Subset(train_dataset, [0])
    #print(f"Dataset size: {len(train_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)
    #train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    time_start = None
    time_finish = None

    model = Demucs(sources=["vocals", "drums", "bass", "other"], resample=False)

    if model_path:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.cuda()
    model.train()

    batches = int(len(train_dataset)/train_loader.batch_size)

    start_epoch = 0
    epochs = 50
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    if model_path:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Starting loaded model from {start_epoch}")

    for epoch in range(start_epoch,epochs):
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
            
            #print(f"Forward split: {(time_forward-time_start if time_start else time_forward):.4f}")
            #print(f"Backward split {(time_backward-time_forward):.4f}")
            if (batch_i % num_workers == 0):
                time_finish = time.time()
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_i}/{batches}, Loss = {loss.item():.4f}, Duration = {(time_finish-time_start if time_start else time_finish):.4f}")
                time_start = time.time()
            
        
        print(f"Completed Epoch {epoch+1}/{epochs} with Loss = {loss.item():.4f}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, f'demucs_20250613_{epoch+1:02}.pth')

if __name__ == "__main__":
    model_path = "demucs_20250613_48.pth"
    train(model_path)