import os
import torch
from torch.optim import Adam
from tqdm import tqdm
from .dataset import get_simclr_dataloader
from .models import ResNet18SimCLR
from .losses import NTXentLoss

def train_simclr(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training SimCLR on {device}")
    
    loader = get_simclr_dataloader(config)
    
    model = ResNet18SimCLR(proj_dim=config['simclr_proj_dim']).to(device)
    optimizer = Adam(model.parameters(), lr=config['simclr_lr'], weight_decay=config['weight_decay'])
    criterion = NTXentLoss(temperature=config['simclr_temperature'])
    
    epochs = config['simclr_epochs']
    
    os.makedirs('checkpoints', exist_ok=True)
    best_loss = float('inf')
    resume_path = 'checkpoints/simclr_resume.pth'
    start_epoch = 0
    if os.path.exists(resume_path):
        print(f"Resuming SimCLR from checkpoint {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"Resumed SimCLR at epoch {start_epoch}")
    
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        
        # We don't have validation set for SimCLR in standard setup unless we use a holdout.
        # So we just train for max epochs and save the last or best training loss checkpoint.
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for (x1, x2), _, _, _ in pbar:
            x1, x2 = x1.to(device), x2.to(device)
            
            optimizer.zero_grad()
            _, z1 = model(x1)
            _, z2 = model(x2)
            
            loss = criterion(z1, z2)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        epoch_loss = running_loss / len(loader)
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({'state_dict': model.state_dict()}, 'checkpoints/simclr_best.pth')
            
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss
        }, resume_path)
            
    print("SimCLR Training completed.")
