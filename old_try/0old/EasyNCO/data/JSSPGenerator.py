import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader



from EasyNCO.utils.utils import getLogger
from EasyNCO.data.LIBUtils import jssp_result

logger = getLogger(__name__)
import re

def JSSPGenerator(data_size, problem_size, batch_size, device='cpu', path=None,**kwargs):
    if problem_size is not None:
        num_job, num_machine = map(int, re.split(r"[×x]", problem_size))

    if path is None:
        dataset = random_jssp_generator(num_sample=data_size,
                                        num_job=num_job,
                                        num_machine=num_machine,
                                        device=device,**kwargs)

    else:
        dataset = customized_jssp_loader(num_sample=data_size,
                                         device=device,
                                         path=path)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0,
                             collate_fn=None)
    return data_loader


class random_jssp_generator(Dataset):
    def __init__(
        self,
        num_sample,
        num_job,
        num_machine,
        device,
        **kwargs
    ):
        time_low = kwargs.get("time_low", 1)
        time_high = kwargs.get("time_high", 99)
        self.num_sample = num_sample
        self.num_machine = num_machine
        self.num_job = num_job
        self.time_low = time_low
        self.time_high = time_high
        self.device = device

    def permute_rows(self,x):
        """
        :param x: np array 2-D
        :return:
        """
        num_rows, num_cols = x.size()
        rand_vals = torch.rand((num_rows, num_cols), device=x.device)
        ix_j = rand_vals.argsort(dim=1)
        ix_i = torch.arange(num_rows, device=x.device).unsqueeze(1).expand(-1, num_cols)

        return x[ix_i, ix_j]

    def __len__(self):
        return self.num_sample

    def __getitem__(self, idx):
        times = torch.randint(
            low=self.time_low,
            high=self.time_high,
            size=(self.num_job, self.num_machine)
        )
        machines = torch.arange(1, self.num_machine + 1).unsqueeze(0).repeat(self.num_job, 1)
        machines = self.permute_rows(machines)

        jsp_instance = torch.cat([times.unsqueeze(0), machines.unsqueeze(0)], dim=0)
        return jsp_instance #  shape [2,job,machine]


class customized_jssp_loader(Dataset):
    def __init__(self, num_sample, device="cpu", path=None):
        self.num_sample = num_sample
        self.device = device
        self.path = path
        self.file_type = os.path.splitext(self.path)[-1]


        if self.file_type == ".npy":
            loaded = np.load(self.path)
            filename = os.path.splitext(os.path.basename(path))[0]
            optimal = jssp_result[filename]
            data = list(zip(loaded, optimal))
            self.data = data[:self.num_sample]
        else:
            raise NotImplementedError(
                f"The file type ({self.file_type}) is not supported in current version"
            )

    def __getitem__(self, index):

        data = torch.from_numpy(self.data[index][0])
        optimal = self.data[index][1]
        return {
            "data": data,
            "optimal": optimal
        }

    def __len__(self):
        return self.num_sample
