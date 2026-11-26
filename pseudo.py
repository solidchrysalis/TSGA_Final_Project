"""Pseudo-anomalies for the one-class inner loop (our contribution).

The FS-OCC support set holds only normal examples, so a plain cross-entropy inner loss is one-class and
degenerate (it just pushes every input towards "normal"). We synthesise anomalies from the support images
themselves, label them 1, and adapt on a two-class support set. Only the support set is used, so this is
valid at meta-test time too; the query set is never touched.

Kinds (all operate on a [K, C, H, W] batch in [0, 1]):
  cutpaste : paste a colour-jittered rectangle cut from a random support image at a random location
             (local defect, after CutPaste, Li et al. CVPR 2021)
  shuffle  : split into a g x g grid and permute the tiles (keeps local statistics, breaks global structure)
  mix      : each pseudo-anomaly uses one of the two at random
"""
import random
import torch


def cutpaste(x, min_frac=0.25, max_frac=0.5, jitter=0.2):
    K, C, H, W = x.shape
    out = x.clone()
    for i in range(K):
        h = random.randint(int(H * min_frac), int(H * max_frac))
        w = random.randint(int(W * min_frac), int(W * max_frac))
        src = x[random.randrange(K)]
        ys, xs = random.randrange(H - h + 1), random.randrange(W - w + 1)
        yd, xd = random.randrange(H - h + 1), random.randrange(W - w + 1)
        scale = 1 + jitter * (2 * torch.rand(C, 1, 1, device=x.device) - 1)
        out[i, :, yd:yd + h, xd:xd + w] = (src[:, ys:ys + h, xs:xs + w] * scale).clamp(0, 1)
    return out


def patch_shuffle(x, g=4):
    K, C, H, W = x.shape
    ph, pw = H // g, W // g
    tiles = x[:, :, :ph * g, :pw * g].reshape(K, C, g, ph, g, pw).permute(0, 2, 4, 1, 3, 5).reshape(K, g * g, C, ph, pw)
    idx = torch.stack([torch.randperm(g * g, device=x.device) for _ in range(K)])
    tiles = tiles[torch.arange(K, device=x.device)[:, None], idx]
    out = x.clone()
    out[:, :, :ph * g, :pw * g] = tiles.reshape(K, g, g, C, ph, pw).permute(0, 3, 1, 4, 2, 5).reshape(K, C, ph * g, pw * g)
    return out


def make_pseudo_anomalies(x, kind):
    if kind == 'cutpaste':
        return cutpaste(x)
    if kind == 'shuffle':
        return patch_shuffle(x)
    if kind == 'mix':
        use_cp = (torch.rand(x.shape[0], device=x.device) < 0.5)[:, None, None, None]
        return torch.where(use_cp, cutpaste(x), patch_shuffle(x))
    raise ValueError('unknown pseudo-anomaly kind: {}'.format(kind))


def auroc(scores, labels):
    """AUROC of anomaly scores (higher = more anomalous) for one task, via the Mann-Whitney U statistic."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return torch.tensor(float('nan'))
    diff = pos[:, None] - neg[None, :]
    return ((diff > 0).float() + 0.5 * (diff == 0).float()).mean()
