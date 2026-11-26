"""Download the FS-OCC datasets via torchmeta and build their hdf5 files (run on a compute node)."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from torchmeta.datasets import MiniImagenet, CIFARFS, FC100, Omniglot, CUB

CLASSES = {"miniimagenet": MiniImagenet, "cifarfs": CIFARFS, "fc100": FC100, "omniglot": Omniglot, "cub": CUB}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--root", default="/scratch/%s/tsga/data" % os.environ.get("USER", "user"))
    a = ap.parse_args()
    bad = [n for n in a.names if n not in CLASSES]
    if bad:
        ap.error("unknown dataset(s) %s; choose from %s" % (bad, list(CLASSES)))
    os.makedirs(a.root, exist_ok=True)
    if "cub" in (a.names or list(CLASSES)):
        # Google Drive often rate-limits the CUB archive; pre-fetch it from Caltech (same file, md5-checked by torchmeta)
        from torchvision.datasets.utils import download_url
        cub_dir = os.path.join(a.root, "cub")
        if not os.path.isfile(os.path.join(cub_dir, "CUB_200_2011.tgz")) and not os.path.isfile(os.path.join(cub_dir, "test_data.hdf5")):
            download_url("https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1",
                         cub_dir, filename="CUB_200_2011.tgz", md5="97eceeb196236b17998738112f37df78")
    for n in (a.names or list(CLASSES)):
        for split in ("meta_train", "meta_val", "meta_test"):
            ds = CLASSES[n](a.root, num_classes_per_task=2, download=True, **{split: True})
            print(n, split, "classes:", ds.dataset.num_classes if hasattr(ds, "dataset") else "?", flush=True)
    print("DOWNLOAD_DONE")
