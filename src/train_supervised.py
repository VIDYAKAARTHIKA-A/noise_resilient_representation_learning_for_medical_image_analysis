import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from .dataset import get_dataloaders
from .models import ResNet18Classifier, build_reference_encoder
from .utils import EarlyStopping, evaluate_model, save_results, set_seed
from .detection import (extract_features, predict_probs, knn_neighbors, neighbor_votes,
                        knn_clean_score, gmm_clean_prob, per_class_auroc)
from .losses import ce_per_sample, gce_per_sample

_V2 = dict(method='v2', use_knn=True, use_gmm=True, relabel=True, logit_adjust=True)

# Every stage is a flag set, so the ablation table is just a list of stage names.
STAGES = {
    'baseline':        dict(init='imagenet', method='ce_weighted'),
    'simclr_finetune': dict(init='simclr',   method='ce_weighted'),
    'v2_full':         dict(init='imagenet', **_V2),
    'v2_no_relabel':   dict(init='imagenet', **{**_V2, 'relabel': False}),
    'v2_no_gmm':       dict(init='imagenet', **{**_V2, 'use_gmm': False}),
    'v2_no_knn':       dict(init='imagenet', **{**_V2, 'use_knn': False, 'relabel': False}),
    'v2_no_logitadj':  dict(init='imagenet', **{**_V2, 'logit_adjust': False}),
    'v2_simclr_init':  dict(init='simclr',   **_V2),   # same init as simclr_finetune -> clean comparison
}

_V3 = dict(method='v2', use_knn=True, use_gmm=True, relabel=True,
           logit_adjust=False, class_balance=True)

STAGES.update({
    # v3 = v2 + class-balanced weights from E_c(t), class-relative kNN threshold,
    #      rare-class protection (no relabel out of rare classes, higher weight floor), CE loss
    'v3_imagenet':     dict(init='imagenet', ref='imagenet', **_V3),
    'v3_simclr':       dict(init='simclr',   ref='simclr', freeze_warmup=True, **_V3),
    'v3s_no_relabel':  dict(init='simclr', ref='simclr', freeze_warmup=True, **{**_V3, 'relabel': False}),
    'v3s_no_gmm':      dict(init='simclr', ref='simclr', freeze_warmup=True, **{**_V3, 'use_gmm': False}),
    'v3s_no_knn':      dict(init='simclr', ref='simclr', freeze_warmup=True, **{**_V3, 'use_knn': False, 'relabel': False}),
    'v3s_no_cb':       dict(init='simclr', ref='simclr', freeze_warmup=True, **{**_V3, 'class_balance': False}),
})

DEFAULT_V2 = dict(cb_beta=1.0, rare_frac=0.05, rare_keep_frac=0.7, rare_w_floor=1.0,
                  knn_rel_thresh=True, ref_encoder='imagenet', knn_k=20, knn_thresh=0.3, knn_temp=0.1, w_knn=0.7,
                  ema=0.7, warmup_epochs=2, nu=10.0, min_keep_frac=0.2, w_floor=0.5,
                  relabel_conf=0.9, relabel_vote=0.5, loss='gce', gce_q=0.7, logit_tau=1.0,
                  refresh_features_every=0, log_every=5)

LOG_HEADER = ('epoch,class,n_labeled,frac_clean,auroc_knn,auroc_gmm,auroc_comb,'
              'mean_w_clean,mean_w_noisy,n_relabeled,relabel_acc\n')


def _log_detection(path, epoch, C, orig, clean_gt, true, work, w, knn_raw, p_gmm, clean_ema, flip):
    """Ground-truth diagnostics (never used for training)."""
    nan = [float('nan')] * C
    au_k = per_class_auroc(knn_raw, clean_gt, orig, C)
    au_g = per_class_auroc(p_gmm, clean_gt, orig, C) if p_gmm is not None else nan
    au_c = per_class_auroc(clean_ema, clean_gt, orig, C)
    with open(path, 'a') as f:
        for c in range(C):
            m = orig == c
            wc, wn = w[m & clean_gt], w[m & ~clean_gt]
            fl = flip & m
            relabel_acc = float((work[fl] == true[fl]).mean()) if fl.any() else float('nan')
            f.write(f"{epoch},{c},{int(m.sum())},{clean_gt[m].mean():.4f},{au_k[c]:.4f},{au_g[c]:.4f},"
                    f"{au_c[c]:.4f},{wc.mean() if len(wc) else float('nan'):.4f},"
                    f"{wn.mean() if len(wn) else float('nan'):.4f},{int(fl.sum())},{relabel_acc:.4f}\n")


def train_supervised(config, noise_level, stage, run_name, noise_type='uniform_global',
                     val_mode='clean', seed=42):
    set_seed(seed)
    spec = STAGES[stage]
    is_v2 = spec['method'] == 'v2'
    use_knn, use_gmm = spec.get('use_knn', False), spec.get('use_gmm', False)
    relabel, use_la = spec.get('relabel', False), spec.get('logit_adjust', False)
    use_cb = spec.get('class_balance', False)
    v2 = {**DEFAULT_V2, **config.get('v2', {})}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{run_name}] stage={stage} noise={noise_level:.0%} type={noise_type} val={val_mode} seed={seed} on {device}")

    data = get_dataloaders(config, noise_level, noise_type, val_mode, seed)
    train_loader, val_loader, test_loader, score_loader = data['train'], data['val'], data['test'], data['score']
    C, train_ds = data['num_classes'], data['train_dataset']
    N = len(train_ds)
    orig = np.asarray(train_ds.labels).astype(np.int64)                 # given (noisy) labels
    clean_gt = np.asarray(train_ds.is_clean).astype(bool)               # analysis only
    true = train_ds.df['original_label'].values.astype(np.int64)        # analysis only
    ar = np.arange(N)
    rare = np.bincount(orig, minlength=C) < v2['rare_frac'] * N          # classes protected from the filter

    model = ResNet18Classifier(C, init=spec['init']).to(device)
    optimizer = Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    static_w = data['class_weights'].to(device)
    loss_fn = (lambda lg, y: gce_per_sample(lg, y, v2['gce_q'])) if v2['loss'] == 'gce' else ce_per_sample

    # ---- frozen reference features (label-agnostic anchor) -------------------------------
    need_nbr = is_v2 and (use_knn or relabel)
    votes = knn_raw = None
    def set_neighbors(feats):
        nonlocal votes, knn_raw
        votes = neighbor_votes(knn_neighbors(feats, v2['knn_k']), orig, C)
        knn_raw = votes[ar, orig]
    if need_nbr:
        ref = build_reference_encoder(spec.get('ref', v2['ref_encoder'])).to(device)
        set_neighbors(extract_features(ref, score_loader, device))
        del ref

    # ---- bookkeeping ---------------------------------------------------------------------
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    ckpt_path, resume_path = f"checkpoints/{run_name}.pth", f"checkpoints/{run_name}_resume.pth"
    log_path = f"logs/{run_name}_detection.csv"
    use_val = val_mode != 'last'
    early = EarlyStopping(patience=config['patience'], mode='max')
    epochs, start_epoch = config['epochs'], 0
    clean_ema = prob_ema = None
    best_adjust = torch.zeros(C, device=device)

    if os.path.exists(resume_path):
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optim'])
        start_epoch = ck['epoch'] + 1
        early.best_score, early.counter = ck['es_best'], ck['es_counter']
        clean_ema, prob_ema, best_adjust = ck['clean_ema'], ck['prob_ema'], ck['best_adjust'].to(device)
        print(f"Resumed at epoch {start_epoch}")
    elif is_v2 and noise_level > 0:
        with open(log_path, 'w') as f:
            f.write(LOG_HEADER)

    adjust_t = torch.zeros(C, device=device)
    labels_t = torch.tensor(orig, device=device)
    sample_w = torch.ones(N, device=device)

    for epoch in range(start_epoch, epochs):
        # ================= E-step: update weights / labels / prior (v2 only) =================
        p_gmm, flip = None, np.zeros(N, dtype=bool)
        if is_v2:
            warm = v2['warmup_epochs']
            scoring = epoch >= warm
            if (need_nbr and scoring and v2['refresh_features_every'] > 0
                    and (epoch - warm) % v2['refresh_features_every'] == 0):
                set_neighbors(extract_features(model.backbone, score_loader, device))

            if scoring and (use_gmm or relabel):
                probs = predict_probs(model, score_loader, device, N, C)
                prob_ema = probs if prob_ema is None else v2['ema'] * prob_ema + (1 - v2['ema']) * probs
                if use_gmm:
                    losses = -np.log(np.clip(probs[ar, orig], 1e-7, 1.0))
                    p_gmm = gmm_clean_prob(losses, orig, nu=v2['nu'], seed=seed)

            p_knn = None
            if use_knn:
                thr = np.full(C, v2['knn_thresh'])
                if v2['knn_rel_thresh']:    # rare classes have few same-label neighbours by construction
                    for c in range(C):
                        if (orig == c).any():
                            thr[c] = max(0.05, min(v2['knn_thresh'], 0.5 * np.median(knn_raw[orig == c])))
                p_knn = knn_clean_score(knn_raw, thr[orig], v2['knn_temp'])
            if p_knn is not None and p_gmm is not None:
                p = v2['w_knn'] * p_knn + (1 - v2['w_knn']) * p_gmm
            elif p_knn is not None:
                p = p_knn
            elif p_gmm is not None:
                p = p_gmm
            else:
                p = np.ones(N)
            clean_ema = p if clean_ema is None else v2['ema'] * clean_ema + (1 - v2['ema']) * p

            # Relabel only when BOTH the (smoothed) model and the frozen-feature neighbours
            # confidently agree on a different class. Votes always use the ORIGINAL labels,
            # so decisions never compound on earlier relabelings.
            work = orig.copy()
            p_eff = clean_ema.copy()
            if relabel and prob_ema is not None:
                pred, conf = prob_ema.argmax(1), prob_ema.max(1)
                vote_arg, vote_conf = votes.argmax(1), votes.max(1)
                flip = ((pred == vote_arg) & (pred != orig)
                        & (conf >= v2['relabel_conf']) & (vote_conf >= v2['relabel_vote']))
                flip &= ~rare[orig]
                work[flip] = pred[flip]
                p_eff[flip] = 0.5 * (conf[flip] + vote_conf[flip])

            # Per-class floor: the top min_keep_frac of every class always keeps a real weight,
            # so no rare class can be wiped out by the detector.
            w = p_eff.copy()
            for c in range(C):
                idx_c = np.where(work == c)[0]
                if len(idx_c):
                    kf, fl = ((v2['rare_keep_frac'], v2['rare_w_floor']) if (use_cb and rare[c])
                              else (v2['min_keep_frac'], v2['w_floor']))
                    k = max(1, int(np.ceil(kf * len(idx_c))))
                    top = idx_c[np.argsort(-w[idx_c])[:k]]
                    w[top] = np.maximum(w[top], fl)

            # E_c(t): clean mass per class actually used for training -> prior for logit adjustment
            E_c = np.bincount(work, weights=w, minlength=C) + 1e-6
            if use_cb:   # imbalance correction driven by the estimated CLEAN mass per class
                cls_w = (E_c.sum() / (C * E_c)) ** v2['cb_beta']
                w = w * cls_w[work]
                w = w / max(w.mean(), 1e-8)
            adjust = v2['logit_tau'] * np.log(E_c / E_c.sum()) if use_la else np.zeros(C)
            adjust_t = torch.tensor(adjust, dtype=torch.float32, device=device)
            sample_w = torch.tensor(w, dtype=torch.float32, device=device)
            labels_t = torch.tensor(work, device=device)

            if noise_level > 0 and (epoch == warm or (epoch - warm) % v2['log_every'] == 0):
                _log_detection(log_path, epoch, C, orig, clean_gt, true, work, w,
                               knn_raw if use_knn else np.zeros(N), p_gmm, clean_ema, flip)

        if spec.get('freeze_warmup'):   # noisy labels cannot touch the label-free features during warmup
            for p_ in model.backbone.parameters():
                p_.requires_grad = epoch >= v2['warmup_epochs']

        # ================= M-step: gradient steps =================
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for inputs, targets, idx, _ in pbar:
            inputs, idx = inputs.to(device), idx.to(device)
            optimizer.zero_grad()
            out = model(inputs)
            if is_v2:
                loss = (sample_w[idx] * loss_fn(out, labels_t[idx])).mean()
            else:
                loss = F.cross_entropy(out, targets.to(device), weight=static_w)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': f'{loss.item():.3f}'})

        # ================= validation / early stopping =================
        if use_val:
            val = evaluate_model(model, val_loader, device, C, logit_adjust=adjust_t if use_la else None)
            print(f"Epoch {epoch + 1} Val Macro F1 ({val_mode}): {val['macro_f1']:.4f}")
            early(val['macro_f1'], model, ckpt_path)
            if early.counter == 0:
                best_adjust = adjust_t.clone()

        torch.save({'epoch': epoch, 'model': model.state_dict(), 'optim': optimizer.state_dict(),
                    'es_best': early.best_score, 'es_counter': early.counter,
                    'clean_ema': clean_ema, 'prob_ema': prob_ema, 'best_adjust': best_adjust.cpu()},
                   resume_path)
        if use_val and early.early_stop:
            print("Early stopping triggered")
            break

    # ================= test =================
    if use_val:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        best_adjust = adjust_t.clone()
        torch.save(model.state_dict(), ckpt_path)

    res = evaluate_model(model, test_loader, device, C, logit_adjust=best_adjust if use_la else None)
    res['macro_f1_unadjusted'] = (evaluate_model(model, test_loader, device, C)['macro_f1']
                                  if use_la else res['macro_f1'])
    res.update(stage=stage, noise_level=noise_level, noise_type=noise_type,
               val_mode=val_mode, seed=seed, run_name=run_name)
    save_results(res, 'results.json')
    if os.path.exists(resume_path):
        os.remove(resume_path)
    print(f"[{run_name}] Test Macro F1: {res['macro_f1']:.4f}  (unadjusted: {res['macro_f1_unadjusted']:.4f})")
    return res