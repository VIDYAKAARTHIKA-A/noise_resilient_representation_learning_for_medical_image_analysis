"""
Label-noise detection utilities.

Two independent witnesses that a label is clean:
  1. kNN agreement in a FROZEN feature space (label-agnostic; does not depend on a
     classifier that is itself biased by noise and imbalance).
  2. Class-wise GMM on the training loss (the original idea), with the shared-variance
     prior and a safeguard for classes whose loss is not bimodal.
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score


# --------------------------------------------------------------------------- extraction
@torch.no_grad()
def extract_features(backbone, loader, device):
    """L2-normalised features, returned in dataset order (loader must give idx)."""
    was_training = backbone.training
    backbone.eval()
    feats, idxs = [], []
    for x, _, i, _ in loader:
        feats.append(F.normalize(backbone(x.to(device)), dim=1).cpu())
        idxs.append(i)
    backbone.train(was_training)
    feats = torch.cat(feats).numpy()
    idxs = torch.cat(idxs).numpy()
    out = np.empty_like(feats)
    out[idxs] = feats
    return out


@torch.no_grad()
def predict_probs(model, loader, device, n_samples, num_classes):
    """Softmax outputs for every training sample, in dataset order (no augmentation)."""
    was_training = model.training
    model.eval()
    probs = np.zeros((n_samples, num_classes), dtype=np.float32)
    for x, _, i, _ in loader:
        probs[i.numpy()] = F.softmax(model(x.to(device)), dim=1).cpu().numpy()
    model.train(was_training)
    return probs


# --------------------------------------------------------------------------- kNN witness
def knn_neighbors(feats, k=20):
    """Indices of the k nearest neighbours of every sample (self excluded)."""
    nn_ = NearestNeighbors(n_neighbors=k + 1, metric='cosine', algorithm='brute', n_jobs=-1).fit(feats)
    idx = nn_.kneighbors(feats, n_neighbors=k + 1, return_distance=False)
    self_mask = idx == np.arange(len(feats))[:, None]
    self_mask[~self_mask.any(axis=1), -1] = True       # self not returned (duplicate images): drop the farthest
    return idx[~self_mask].reshape(len(feats), k)


def neighbor_votes(nbr, labels, num_classes):
    """votes[i, c] = fraction of i's neighbours carrying label c."""
    nl = labels[nbr]
    votes = np.zeros((len(labels), num_classes), dtype=np.float64)
    for c in range(num_classes):
        votes[:, c] = (nl == c).mean(axis=1)
    return votes


def knn_clean_score(agreement, thresh=0.3, temp=0.1):
    """Soft threshold on neighbour agreement: ~1 when >= half the neighbours agree, ~0 when almost none."""
    return 1.0 / (1.0 + np.exp(-(agreement - thresh) / temp))


# --------------------------------------------------------------------------- GMM witness
def gmm_clean_prob(losses, targets, nu=10.0, seed=42):
    """
    Class-wise 2-component GMM on min-max normalised loss with the empirical-Bayes
    shared-variance prior. If a class's two components are barely separated (no real
    bimodality, e.g. a clean class), the posterior is shrunk toward 1 instead of
    arbitrarily calling half the class noisy.
    """
    losses = np.asarray(losses, dtype=np.float64)
    targets = np.asarray(targets)
    norm = (losses - losses.min()) / (losses.max() - losses.min() + 1e-8)
    global_var = np.var(norm)
    prob = np.full(len(norm), 0.5)

    for c in np.unique(targets):
        m = targets == c
        x = norm[m].reshape(-1, 1)
        n_c = len(x)
        if n_c < 10:
            prob[m] = 1.0
            continue
        try:
            gmm = GaussianMixture(n_components=2, max_iter=100, tol=1e-3,
                                  reg_covar=5e-4, random_state=seed).fit(x)
        except Exception:
            prob[m] = 0.5
            continue
        raw = gmm.covariances_.flatten()
        adj = (n_c * raw + nu * global_var) / (n_c + nu)
        gmm.covariances_ = adj.reshape(-1, 1, 1)
        gmm.precisions_cholesky_ = (1.0 / np.sqrt(adj)).reshape(-1, 1, 1)   # predict_proba uses this
        post = gmm.predict_proba(x)[:, int(gmm.means_.argmin())]

        sep = abs(gmm.means_[0, 0] - gmm.means_[1, 0]) / np.sqrt(adj.mean())
        trust = float(np.clip(sep / 3.0, 0.0, 1.0))
        prob[m] = trust * post + (1.0 - trust) * 1.0
    return prob


# --------------------------------------------------------------------------- diagnostics
def per_class_auroc(score, is_clean, targets, num_classes):
    """AUROC of 'score predicts is_clean' inside each (noisy) class. NaN if undefined."""
    out = []
    for c in range(num_classes):
        m = np.asarray(targets) == c
        y = np.asarray(is_clean)[m].astype(int)
        if m.sum() < 2 or y.min() == y.max():
            out.append(float('nan'))
        else:
            out.append(float(roc_auc_score(y, np.asarray(score)[m])))
    return out
