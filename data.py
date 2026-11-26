"""Few-shot OCC task distributions from torchmeta hdf5 files; sampling logic from OC-MAML (github.com/AhmedFrikha/Few-Shot-One-Class-Classification-via-Meta-Learning)."""
import json
import os

import h5py
import numpy as np

from task import CB_OCCTask, OCCTask
from task_distribution import TaskDistribution

# args.dataset key -> (folder, layout)
DATASETS = {
    'MIN': ('miniimagenet', 'min'),
    'CIFAR': ('cifar100', 'cifarfs'),
    'FC': ('cifar100', 'fc100'),
    'OMN': ('omniglot', 'omn'),
    'CUB': ('cub', 'cub'),       # extension: CUB-200-2011, torchmeta 100/50/50 class split (not in the paper)
}


def default_root():
    return os.environ.get('TSGA_DATA', '/scratch/{}/tsga/data'.format(os.environ.get('USER', 'user')))


def _resize(x, size):
    # uint8 [N,H,W,C] -> [N,size,size,C], antialiased bilinear
    import torch
    import torch.nn.functional as F
    out = []
    for i in range(0, len(x), 1024):
        t = torch.from_numpy(x[i:i + 1024]).permute(0, 3, 1, 2).float()
        t = F.interpolate(t, size=(size, size), mode='bilinear', align_corners=False, antialias=True)
        out.append(t.round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy())
    return np.concatenate(out)


def load_split(root, dataset, split, image_size=None):
    """Load one meta-split (train|val|test) into RAM as float32 [N,C,H,W] in [0,1] plus int64 labels."""
    folder, kind = DATASETS[dataset]
    base = os.path.join(root, folder)
    xs, ys = [], []
    if kind == 'min':
        labels = json.load(open(os.path.join(base, '{}_labels.json'.format(split))))
        with h5py.File(os.path.join(base, '{}_data.hdf5'.format(split)), 'r') as f:
            for i, c in enumerate(labels):
                a = f['datasets'][c][()]
                xs.append(a); ys.append(np.full(len(a), i))
    elif kind in ('cifarfs', 'fc100'):
        sub = 'cifar-fs' if kind == 'cifarfs' else 'fc100'
        labels = json.load(open(os.path.join(base, sub, '{}_labels.json'.format(split))))
        with h5py.File(os.path.join(base, 'data.hdf5'), 'r') as f:
            for i, (coarse, fine) in enumerate(labels):
                a = f[coarse][fine][()]
                xs.append(a); ys.append(np.full(len(a), i))
    elif kind == 'omn':
        labels = json.load(open(os.path.join(base, 'vinyals_{}_labels.json'.format(split))))
        with h5py.File(os.path.join(base, 'data.hdf5'), 'r') as f:
            for i, (grp, alphabet, character) in enumerate(labels):
                a = f[grp][alphabet][character][()][..., None]   # [20,105,105] -> [20,105,105,1]
                xs.append(a); ys.append(np.full(len(a), i))
    elif kind == 'cub':
        # torchmeta stores CUB as variable-size JPEG bytes; decode and resize each image here
        import io
        from PIL import Image
        size = image_size or 84
        labels = json.load(open(os.path.join(base, '{}_labels.json'.format(split))))
        with h5py.File(os.path.join(base, '{}_data.hdf5'.format(split)), 'r') as f:
            for i, c in enumerate(labels):
                imgs = [np.asarray(Image.open(io.BytesIO(b.tobytes())).convert('RGB')
                                   .resize((size, size), Image.BILINEAR)) for b in f['datasets'][c][()]]
                xs.append(np.stack(imgs)); ys.append(np.full(len(imgs), i))
    else:
        raise ValueError(kind)
    X, y = np.concatenate(xs), np.concatenate(ys).astype(np.int64)
    if image_size is not None and X.shape[1] != image_size:
        X = _resize(X, image_size)
    X = X.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    return X, y


def create_task_distributions(
    root,
    dataset,
    train_occ,
    test_occ,
    num_training_samples_per_class=10,
    num_test_samples_per_class=15,
    num_training_classes=20,
    meta_batch_size=5,
    seq_length=0,
    image_size=None,
):
    """Upstream-compatible create_*_task_distribution; returns (metatrain, metaval, metatest)."""
    root = root or default_root()
    trainX, trainY = load_split(root, dataset, 'train', image_size)
    valX, valY = load_split(root, dataset, 'val', image_size)
    testX, testY = load_split(root, dataset, 'test', image_size)

    def make(X, Y, occ, n_test):
        cls = OCCTask if occ else CB_OCCTask
        return TaskDistribution(
            tasks=[cls(X, Y, num_training_samples_per_class, n_test, num_training_classes,
                       split_train_test=0.5)],
            task_probabilities=[1.0],
            batch_size=meta_batch_size,
            sample_with_replacement=True,
        )

    return (make(trainX, trainY, train_occ, num_test_samples_per_class),
            make(valX, valY, test_occ, num_test_samples_per_class),
            make(testX, testY, test_occ, num_test_samples_per_class))


# Names used by utils.OCCData (first arg = data root folder).
def create_miniimagenet_task_distribution(root, train_occ, test_occ, **kw):
    return create_task_distributions(root, 'MIN', train_occ, test_occ, **kw)


def create_cifarfs_task_distribution(root, train_occ, test_occ, **kw):
    return create_task_distributions(root, 'CIFAR', train_occ, test_occ, **kw)


def create_fc100_task_distribution(root, train_occ, test_occ, **kw):
    return create_task_distributions(root, 'FC', train_occ, test_occ, **kw)


def create_omniglot_allcharacters_task_distribution(root, train_occ, test_occ, **kw):
    return create_task_distributions(root, 'OMN', train_occ, test_occ, **kw)
