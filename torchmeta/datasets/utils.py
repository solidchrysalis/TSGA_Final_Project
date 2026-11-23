import os
import json


def get_asset_path(*args):
    basedir = os.path.dirname(__file__)
    return os.path.join(basedir, 'assets', *args)


def get_asset(*args, dtype=None):
    filename = get_asset_path(*args)
    if not os.path.isfile(filename):
        raise IOError('{} not found'.format(filename))

    if dtype is None:
        _, dtype = os.path.splitext(filename)
        dtype = dtype[1:]

    if dtype == 'json':
        with open(filename, 'r') as f:
            data = json.load(f)
    else:
        raise NotImplementedError()
    return data

# Local edit: Drive downloads go through a wrapper that validates the file id, skips existing files, then calls torchvision's gdown-based downloader.
import re
from torchvision.datasets.utils import check_integrity
from torchvision.datasets.utils import download_file_from_google_drive as _tv_download_file_from_google_drive

_GDRIVE_ID_RE = re.compile(r'^[A-Za-z0-9_-]{10,}$')


def _gdrive_file_id(file_id):
    """Return file_id only if it is just the id portion of a Drive link."""
    if not isinstance(file_id, str) or not _GDRIVE_ID_RE.match(file_id):
        raise ValueError('Invalid Google Drive file id {0!r}: expected only the id portion of a '
                         'Drive link ([A-Za-z0-9_-])'.format(file_id))
    return file_id


def download_file_from_google_drive(file_id, root, filename=None, md5=None):
    """Download a Drive file into `root` unless already present (id validated, md5 checked)."""
    file_id = _gdrive_file_id(file_id)
    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename or file_id)
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
        return fpath
    _tv_download_file_from_google_drive(file_id, root, filename, md5)
    return fpath
