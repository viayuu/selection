import os
import torch
from torch.utils.data import Dataset, DataLoader

from EasyNCO.utils.utils import getLogger

logger = getLogger(__name__)


####################################
# DATA
####################################
def FFSPGenerator(
    data_size,
    problem_size=None,
    batch_size=50,
    device="cpu",
    path=None,
    machine_cnt_list=[4,4,4],
    job_cnt=20,
    time_low=2,
    time_high=10,
    **kwargs
):
    if path is None:
        dataset = random_ffsp_generator(
            num_sample=data_size,
            machine_cnt_list=machine_cnt_list,
            job_cnt=job_cnt,
            time_low=time_low,
            time_high=time_high,
            device=device,
        )
    else:
        dataset = customized_ffsp_loader(
            num_sample=data_size, device=device, path=path
        )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=None,
    )
    return data_loader

class random_ffsp_generator(Dataset):
    def __init__(
        self,
        num_sample,
        machine_cnt_list,
        job_cnt,
        time_low,
        time_high,
        device="cpu",
    ):
        self.num_sample = num_sample
        self.machine_cnt_list = machine_cnt_list
        self.job_cnt = job_cnt
        self.time_low = time_low
        self.time_high = time_high
        self.device = device

    def __len__(self):
        return self.num_sample

    def __getitem__(self, idx):
        return torch.randint(
            low=self.time_low,
            high=self.time_high,
            size=(self.job_cnt, sum(self.machine_cnt_list)),
            device=self.device,
        )


class customized_ffsp_loader(Dataset):
    def __init__(self, num_sample, device="cpu", path=None):
        self.num_sample = num_sample
        self.device = device
        self.path = path
        self.file_type = os.path.splitext(self.path)[-1]

        if self.file_type == ".pt":
            self.data = torch.load(self.path, map_location=self.device).float() # float32

        else:
            raise NotImplementedError(
                f"The file type ({self.file_type}) is not supported in current version"
            )

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_sample
