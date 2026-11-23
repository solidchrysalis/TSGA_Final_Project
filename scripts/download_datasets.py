"""Download the FS-OCC datasets via torchmeta and build their hdf5 files (run on a compute node)."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from torchmeta.datasets import MiniImagenet, CIFARFS, FC100, Omniglot

CLASSES = {"miniimagenet": MiniImagenet, "cifarfs": CIFARFS, "fc100": FC100, "omniglot": Omniglot}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--root", default="/scratch/%s/tsga/data" % os.environ.get("USER", "user"))
    a = ap.parse_args()
    bad = [n for n in a.names if n not in CLASSES]
    if bad:
        ap.error("unknown dataset(s) %s; choose from %s" % (bad, list(CLASSES)))
    os.makedirs(a.root, exist_ok=True)
    for n in (a.names or list(CLASSES)):
        for split in ("meta_train", "meta_val", "meta_test"):
            ds = CLASSES[n](a.root, num_classes_per_task=2, download=True, **{split: True})
            print(n, split, "classes:", ds.dataset.num_classes if hasattr(ds, "dataset") else "?", flush=True)
    print("DOWNLOAD_DONE")
