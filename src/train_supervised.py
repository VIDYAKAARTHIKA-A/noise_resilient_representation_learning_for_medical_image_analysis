import os
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from .dataset import get_dataloaders
from .models import ResNet18Classifier
from .utils import EarlyStopping, evaluate_model, save_results
from .gmm import estimate_clean_probabilities
from .losses import SymmetricCrossEntropy, FocalLoss

def train_supervised(config, noise_level, stage, run_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{run_name}] Training stage '{stage}' at {noise_level*100}% noise on {device}")
    
    train_loader, val_loader, test_loader, num_classes, class_weights, train_dataset = get_dataloaders(config, noise_level)
    class_weights = class_weights.to(device)
    
    pretrained_path = None
    if stage != 'baseline':
        pretrained_path = 'checkpoints/simclr_best.pth'
        
    model = ResNet18Classifier(num_classes=num_classes, pretrained_encoder_path=pretrained_path).to(device)
    optimizer = Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    # Base criterion
    if stage == 'full_proposed':
        # Class balanced and robust loss
        # Use SCE for robust loss. focal loss could also be used.
        criterion = SymmetricCrossEntropy(alpha=1.0, beta=1.0, num_classes=num_classes)
    else:
        # Standard CE
        criterion = nn.CrossEntropyLoss(reduction='none') # For reweighting
        
    epochs = config['epochs']
    early_stopping = EarlyStopping(patience=config['patience'], mode='max')
    checkpoint_path = f"checkpoints/{run_name}.pth"
    resume_path = f"checkpoints/{run_name}_resume.pth"
    
    start_epoch = 0
    if os.path.exists(resume_path):
        print(f"Resuming from checkpoint {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        early_stopping.best_score = checkpoint['early_stopping_best_score']
        early_stopping.counter = checkpoint['early_stopping_counter']
        print(f"Resumed at epoch {start_epoch}")
    
    # Curriculum schedule
    curriculum_epochs = epochs // 2 if 'curriculum' in stage or stage == 'full_proposed' else 0
    
    # GMM estimation logic
    idx_to_clean_prob = None
    if 'gmm' in stage or stage == 'full_proposed':
        print("Estimating GMM probabilities...")
        idx_to_clean_prob, _, _, _ = estimate_clean_probabilities(model, train_loader, device)
        
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        
        # Calculate current keep threshold for curriculum
        if curriculum_epochs > 0:
            # Gradually increase threshold from 0 to 0.5
            threshold = min(0.5, 0.5 * (epoch / curriculum_epochs))
        else:
            threshold = 0.0
            
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, targets, indices, _ in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            indices = indices.numpy()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            if stage == 'full_proposed':
                loss_per_sample = criterion(outputs, targets)
                # Apply class weights manually since SCE doesn't take it directly
                cw = class_weights[targets]
                loss_per_sample = loss_per_sample * cw
            else:
                loss_per_sample = criterion(outputs, targets)
            
            # Apply GMM / Curriculum weights
            if idx_to_clean_prob is not None:
                probs = torch.tensor([idx_to_clean_prob.get(int(idx.item()), 0.5) for idx in indices]).to(device)
                
                if curriculum_epochs > 0:
                    # Hard filtering based on curriculum threshold
                    mask = (probs >= threshold).float()
                    loss_per_sample = loss_per_sample * mask
                else:
                    # Soft reweighting
                    loss_per_sample = loss_per_sample * probs
                    
            loss = loss_per_sample.mean()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        # Validation
        val_results = evaluate_model(model, val_loader, device, num_classes)
        val_macro_f1 = val_results['macro_f1']
        print(f"Epoch {epoch+1} Val Macro F1: {val_macro_f1:.4f}")
        
        early_stopping(val_macro_f1, model, checkpoint_path)
        
        # Re-estimate GMM if needed (we can do this every epoch or once. For simplicity and stability, we'll keep the initial estimation or re-estimate every N epochs)
        # Often GMM is re-estimated as model trains. Let's do it every epoch if GMM is active.
        if ('gmm' in stage or stage == 'full_proposed') and epoch < epochs - 1:
             idx_to_clean_prob, _, _, _ = estimate_clean_probabilities(model, train_loader, device)
             
        # Save resume checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'early_stopping_best_score': early_stopping.best_score,
            'early_stopping_counter': early_stopping.counter
        }, resume_path)
        
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
            
    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(checkpoint_path))
    test_results = evaluate_model(model, test_loader, device, num_classes)
    
    test_results['stage'] = stage
    test_results['noise_level'] = noise_level
    test_results['run_name'] = run_name
    
    # Save results
    save_results(test_results, 'results.json')
    print(f"Test Macro F1: {test_results['macro_f1']:.4f}")
    
    return test_results
