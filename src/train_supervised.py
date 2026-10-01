import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from .dataset import get_dataloaders
from .models import ResNet18Classifier
from .utils import EarlyStopping, evaluate_model, save_results
from .gmm import estimate_clean_probabilities
from .curriculum import run_class_quantile_curriculum
from .losses import UnifiedDynamicHybridLoss


def _log_selection(run_name, epoch, train_dataset, all_targets, all_indices,
                   selected_indices, num_classes):
    """
    Diagnostic: how clean is the set the curriculum selected, per class?
    Uses the ground-truth is_clean flags that the noise injection stored, so it
    only exists for analysis (it is never used for training).
    precision = fraction of selected samples whose label is actually clean
    recall    = fraction of all clean samples of that class that were selected
    Appends rows to selection_log.csv.
    """
    clean_all = getattr(train_dataset, 'is_clean', None)
    if clean_all is None:
        return
    try:
        clean_flag = np.asarray(clean_all).astype(int)[all_indices]
        selected = np.isin(all_indices, np.asarray(selected_indices))
        path = 'selection_log.csv'
        new_file = not os.path.exists(path)
        with open(path, 'a') as f:
            if new_file:
                f.write('run_name,epoch,class,labeled_total,kept,precision,recall\n')
            for c in range(num_classes):
                in_class = (all_targets == c)
                kept = in_class & selected
                n_kept = int(kept.sum())
                precision = float(clean_flag[kept].mean()) if n_kept > 0 else float('nan')
                recall = float(clean_flag[kept].sum() / max(1, clean_flag[in_class].sum()))
                f.write(f"{run_name},{epoch},{c},{int(in_class.sum())},{n_kept},"
                        f"{precision:.4f},{recall:.4f}\n")
    except Exception as e:
        print(f"[warning] selection logging skipped: {e}")


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
    # PART A: Setup cost objectives based on stage
    # -------------------------------------------------------------------------
    is_proposed = (stage == 'full_proposed')

    if is_proposed:
        # Dynamic, closed-loop hybrid loss (Focal + SCE scaled by clean-effective counts)
        criterion = UnifiedDynamicHybridLoss(num_classes=num_classes).to(device)
    else:
        # baseline / simclr_finetune: class-weighted cross entropy
        static_class_weights = static_class_weights.to(device)

    epochs = config['epochs']

    # Schedule of the proposed method (can be overridden in config.json)
    warmup_epochs = config.get('warmup_epochs', 3)            # plain training before any selection
    curriculum_epochs = config.get('curriculum_epochs', 40)   # epochs over which Q(t) grows to Q_end

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

        # ---------------------------------------------------------------------
        # PART B: Expectation step (GMM -> class-quantile filter -> E_c weights)
        # Skipped during the warm-up epochs: an untrained classifier head gives
        # every sample about the same loss, so the first selection would be random.
        # ---------------------------------------------------------------------
        if is_proposed and epoch >= warmup_epochs:
            # estimate_clean_probabilities returns:
            #   (idx_to_prob, raw_losses, clean_prob, targets, indices)
            # clean_prob is the THIRD element.
            _, _, clean_prob, all_targets, all_indices = estimate_clean_probabilities(
                model, train_loader, device, nu=10.0
            )

            selected_indices = run_class_quantile_curriculum(
                all_targets=all_targets,
                all_clean_probs=clean_prob,
                all_indices=all_indices,
                current_epoch=epoch - warmup_epochs,
                total_epochs=curriculum_epochs,
                Q_start=0.2,
                alpha_pacing=1.0
            )

            criterion.update_dynamic_weights(
                selected_indices=selected_indices,
                all_targets=all_targets,
                all_clean_probs=clean_prob,
                all_indices=all_indices
            )

            # Diagnostic log (every 5 epochs of the curriculum)
            if (epoch - warmup_epochs) % 5 == 0:
                _log_selection(run_name, epoch, train_dataset, all_targets,
                               all_indices, selected_indices, num_classes)

            active_subset = Subset(train_loader.dataset, selected_indices)
            active_loader = DataLoader(
                active_subset,
                batch_size=train_loader.batch_size,
                shuffle=True,
                num_workers=train_loader.num_workers,
                pin_memory=train_loader.pin_memory if hasattr(train_loader, 'pin_memory') else False
            )
        else:
            active_loader = train_loader

        # Make sure we train in train mode (scoring the data switches to eval mode)
        model.train()

        # ---------------------------------------------------------------------
        # PART C: Gradient step
        # ---------------------------------------------------------------------
        pbar = tqdm(active_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, targets, _, _ in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            if is_proposed:
                loss = criterion(outputs, targets)
            else:
                loss = F.cross_entropy(outputs, targets, weight=static_class_weights)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        # Validation
        val_results = evaluate_model(model, val_loader, device, num_classes)
        val_macro_f1 = val_results['macro_f1']
        print(f"Epoch {epoch+1} Val Macro F1: {val_macro_f1:.4f}")

        early_stopping(val_macro_f1, model, checkpoint_path)

        # The proposed method adds data over the curriculum phase, so a flat validation
        # curve there is expected. Do not let early stopping end the run before the
        # curriculum has finished; patience starts counting afterwards.
        if is_proposed and epoch < warmup_epochs + curriculum_epochs:
            early_stopping.early_stop = False
            early_stopping.counter = 0

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