import csv
import datetime
import os
import random

import numpy as np
import torch


def seed_everything(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def augment_xy_by_8_fold(problems):
    x = problems[:, :, [0]]
    y = problems[:, :, [1]]

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    return torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)


def resolve_path(base_dir, path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def make_run_dir(root_dir, experiment_name):
    ts = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
    stamp = ts.strftime("%m%d-%H%M%S")
    run_dir = os.path.join(root_dir, f"{experiment_name}-{stamp}")
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


class csv_logger:
    def __init__(self, log_dir, log_name="log.csv"):
        self.file_name = os.path.join(log_dir, log_name)
        self.file_obj = open(self.file_name, "w", newline="")
        self.file_logger = None

    def write(self, log_dict):
        if self.file_logger is None:
            self.file_logger = csv.DictWriter(self.file_obj, fieldnames=list(log_dict.keys()))
            self.file_logger.writeheader()
        self.file_logger.writerow(log_dict)
        self.file_obj.flush()

