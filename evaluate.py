"""Re-evaluate saved runs on fresh meta-test tasks: accuracy and AUROC, with 95% CIs.

Usage: python evaluate.py <run_dir> [<run_dir> ...]   (each run_dir holds .hydra/config.yaml and best_model.pt)
Writes <run_dir>/eval_summary_statistics.csv. Uses TSGA_DATA, if set, as the data folder (runs may move).
"""
import os
import sys
import torch
from omegaconf import OmegaConf
from utils import set_device, set_seed, save_statistics
from tsga import Experiment


def evaluate_run(run_dir, n_tasks=None, seed=0, which='best'):
    args = OmegaConf.merge(OmegaConf.load(os.path.join(os.path.dirname(__file__), 'conf', 'config.yaml')),
                           OmegaConf.load(os.path.join(run_dir, '.hydra', 'config.yaml')))
    OmegaConf.set_struct(args, False)
    args.pop('hydra', None)
    if os.environ.get('TSGA_DATA'):
        args.folder = os.environ['TSGA_DATA']
    n_tasks = n_tasks or args.n_test_episodes
    device = set_device(args.device)
    set_seed(seed)
    exp = Experiment(args, device)
    exp.load_state_dict(torch.load(os.path.join(run_dir, '{}_model.pt'.format(which)), map_location=device))
    acc, acc_std, auc, auc_std, t = exp.evaluate(n_tasks, exp.data.get_test_episode)
    ci = lambda std: 1.96 * std / n_tasks ** 0.5
    row = {'run': os.path.basename(os.path.normpath(run_dir)), 'model': which, 'n_tasks': n_tasks,
           'acc': '{:.2f}'.format(acc), 'acc_ci95': '{:.2f}'.format(ci(acc_std)),
           'auroc': '{:.2f}'.format(auc), 'auroc_ci95': '{:.2f}'.format(ci(auc_std))}
    out = os.path.join(run_dir, 'eval_summary_statistics.csv')
    save_statistics(list(row.keys()), filename=out, create=True)
    save_statistics(list(row.values()), filename=out)
    print('{run}: acc {acc} ± {acc_ci95}, AUROC {auroc} ± {auroc_ci95} ({n_tasks} tasks)'.format(**row), flush=True)
    return row


if __name__ == '__main__':
    for d in sys.argv[1:]:
        evaluate_run(d)
