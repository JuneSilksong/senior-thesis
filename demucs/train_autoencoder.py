import numpy as np
np.float_ = np.float64
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import time
import math

from demucs import Demucs
from musdb18 import MUSDB18_Denoising
from loss import si_sdr_loss, mrstft_loss

def train(model, dataloader, optimizer, config, epoch):
    batch_i = 0
    running = 0
    time_start = time.time()

    for clean, noisy in dataloader:
        clean = clean.cuda()
        noisy = noisy.cuda()

        clean = clean / (clean.abs().max(dim=-1, keepdim=True)[0] + 1e-8)
        noisy = noisy / (noisy.abs().max(dim=-1, keepdim=True)[0] + 1e-8)

        est_clean = model(noisy)

        loss_sisdr = si_sdr_loss(est_clean, clean)
        loss_stft = mrstft_loss(est_clean, clean)
        loss = loss_sisdr + 0.5 * loss_stft

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        batch_i += 1
        running += loss.item()
        
        if (batch_i % config['num_workers'] == 0):
            time_finish = time.time()
            print(f"Epoch {epoch+1}/{config['epochs']}, Batch {batch_i}/{config['batches']}, Running Loss = {loss.item():.4f}, Duration = {(time_finish-time_start):.4f}")
            time_start = time.time()
    
    return running / len(dataloader)

def main():
    model_date = '20250925' #yyyymmdd
    model_epoch = None #'00' # 2sf
    model_path = None #f'/home/user/Github/senior-thesis/demucs_{model_date}_{model_epoch}.pth'
    num_workers = 16

    dataset = MUSDB18_Denoising(split="train")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)
    print(f"Loaded {len(dataset)} training tracks.")

    batches = int(len(dataset)/dataloader.batch_size)

    model = Demucs(sources=["mix"])
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    start_epoch = 0
    epochs = 50
    best_loss = math.inf

    if model_path:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Starting loaded model from epoch {start_epoch}")

    model.cuda()
    model.train()

    config = {
        'epochs': epochs,
        'batches': batches,
        'num_workers': num_workers
    }

    print(f"Completed Epoch {epoch+1}/{epochs} with Loss = {loss:.4f}")
    for epoch in range(start_epoch, epochs):
        loss = train(model, dataloader, optimizer, config, epoch)
        scheduler.step(loss)
        if loss < best_loss:
            best_loss = loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, f'demucs_{model_date}_{epoch+1:02}.pth')
            print(f"Saved demucs_{model_date}_{epoch+1:02}.pth with new best loss of {best_loss}.")

if __name__ == "__main__":
    main()
