import numpy as np


def run_class_quantile_curriculum(all_targets, all_clean_probs, all_indices,
                                  current_epoch, total_epochs,
                                  Q_start=0.2, alpha_pacing=1.0, Q_end=1.0):
    """
    Class-wise quantile curriculum.

    For every class, samples are ranked by their GMM clean probability and only
    the top Q(t) fraction is kept. Q(t) grows from Q_start to Q_end over training:

        Q(t) = Q_start + (Q_end - Q_start) * (t / (T - 1)) ** alpha_pacing

    Selecting per class (rather than globally) keeps minority classes from being
    wiped out by the filter. At least one sample per class is always kept.

    Args:
        all_targets:     (N,) noisy labels, aligned with all_clean_probs / all_indices.
        all_clean_probs: (N,) probability that each sample's label is clean.
        all_indices:     (N,) dataset indices of the samples.
        current_epoch:   0-based epoch number.
        total_epochs:    total number of epochs.
        Q_start:         fraction kept per class at epoch 0.
        alpha_pacing:    pacing exponent (1.0 = linear, >1 = slower start).
        Q_end:           fraction kept per class at the last epoch.

    Returns:
        Sorted list of selected dataset indices (python ints).
    """
    all_targets = np.asarray(all_targets)
    all_clean_probs = np.asarray(all_clean_probs)
    all_indices = np.asarray(all_indices)

    progress = current_epoch / max(1, total_epochs - 1)
    progress = min(max(progress, 0.0), 1.0)
    Q_t = Q_start + (Q_end - Q_start) * (progress ** alpha_pacing)

    selected = []
    for c in np.unique(all_targets):
        class_pos = np.where(all_targets == c)[0]
        n_keep = max(1, int(np.ceil(Q_t * len(class_pos))))
        # Highest clean probability first
        ranked = class_pos[np.argsort(-all_clean_probs[class_pos], kind='stable')]
        selected.extend(all_indices[ranked[:n_keep]].tolist())

    return sorted(int(i) for i in selected)