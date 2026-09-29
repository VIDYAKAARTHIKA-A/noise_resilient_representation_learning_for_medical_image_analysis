import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from .dataset import get_dataloaders
from .models import ResNet18Classifier
from .utils import EarlyStopping, evaluate_model, save_results
from .gmm import estimate_clean_probabilities
from .losses import UnifiedDynamicHybridLoss  # Import the new unified loss engine

def train_supervised(config, noise_level, stage, run_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{run_name}] Training stage '{stage}' at {noise_level*100}% noise on {device}")
    
    train_loader, val_loader, test_loader, num_classes, static_class_weights, train_dataset = get_dataloaders(config, noise_level)
    
    pretrained_path = None
    if stage != 'baseline':
        pretrained_path = 'checkpoints/simclr_best.pth'
        
    model = ResNet18Classifier(num_classes=num_classes, pretrained_encoder_path=pretrained_path).to(device)
    optimizer = Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    # -------------------------------------------------------------------------
    # PART A: Setup Cost Objectives based on Stage Logic
    # -------------------------------------------------------------------------
    is_proposed = (stage == 'full_proposed')
    
    if is_proposed:
        # Initialize the dynamic, closed-loop hybrid loss system
        criterion = UnifiedDynamicHybridLoss(num_classes=num_classes).to(device)
    else:
        # Standard Cross Entropy baseline setup matching original intent
        criterion = nn.CrossEntropyLoss(reduction='mean')
        static_class_weights = static_class_weights.to(device)
        
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
        
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        
        # -------------------------------------------------------------------------
        # PART B: Expectation Step (Closed-Loop Data & Weight Partitioning)
        # -------------------------------------------------------------------------
        if is_proposed:
            # 1. Run the hierarchical class-wise GMM matching your updated signature
            _, clean_prob, _, all_targets, all_indices = estimate_clean_probabilities(
                model, train_loader, device, nu=10.0
            )
            
            # 2. Extract balanced indices via class-quantile filtering adjustments
            selected_indices = run_class_quantile_curriculum(
                all_targets=all_targets, 
                all_clean_probs=clean_prob, 
                all_indices=all_indices,
                current_epoch=epoch, 
                total_epochs=epochs, 
                Q_start=0.2, 
                alpha_pacing=1.0
            )
            
            # 3. Synchronize inverse effective size scaling parameters inside the engine
            criterion.update_dynamic_weights(
                selected_indices=selected_indices, 
                all_targets=all_targets, 
                all_clean_probs=clean_prob, 
                all_indices=all_indices
            )
            
            # 4. Generate the isolated subset loader to guarantee gradient optimization safety
            active_subset = Subset(train_loader.dataset, selected_indices)
            active_loader = DataLoader(
                active_subset, 
                batch_size=train_loader.batch_size, 
                shuffle=True,
                num_workers=train_loader.num_workers,
                pin_memory=train_loader.pin_memory if hasattr(train_loader, 'pin_memory') else False
            )
        else:
            # Baseline execution paths use standard unweighted dataloader arrays
            active_loader = train_loader
            
        # -------------------------------------------------------------------------
        # PART C: Gradient Execution Step
        # -------------------------------------------------------------------------
        pbar = tqdm(active_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, targets, _, _ in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            if is_proposed:
                # The dynamic weight tensor maps directly to inputs internally inside forward pass
                loss = criterion(outputs, targets)
            else:
                # Standard Cross Entropy baseline computation path
                loss = F.cross_entropy(outputs, targets, weight=static_class_weights)
                
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        # Validation evaluation runs
        val_results = evaluate_model(model, val_loader, device, num_classes)
        val_macro_f1 = val_results['macro_f1']
        print(f"Epoch {epoch+1} Val Macro F1: {val_macro_f1:.4f}")
        
        early_stopping(val_macro_f1, model, checkpoint_path)
        
        # Save resume checkpoint state indices
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
