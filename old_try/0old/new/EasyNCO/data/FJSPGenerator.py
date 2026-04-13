import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
from torch.nn.utils.rnn import pad_sequence
from EasyNCO.utils.utils import getLogger
from EasyNCO.neural_solvers.methods.drl_hgnn.utils import nums_detec
from EasyNCO.data.LIBUtils import brandimarte_cost,e_la_cost,r_la_cost,v_la_cost
logger = getLogger(__name__)


def FJSPGenerator(data_size, problem_size, batch_size,device='cpu', path=None,num_job=10, num_machine=5, time_low=1, time_high=20,**kwargs):
    if path is None:
        dataset = random_fjsp_generator(num_sample=data_size,
                                        num_job=num_job,
                                        num_machine=num_machine,
                                        time_low=time_low,
                                        time_high=time_high,
                                        device=device,**kwargs)
    elif path.endswith('public_fjsp_dataset'):
        dataset = public_fjsp_loader(device=device,
                                path=path,
                                **kwargs)
    else:
        dataset = customized_fjsp_loader(num_sample=data_size,
                                         device=device,
                                         path=path)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0,
                             collate_fn=None)
    return data_loader


class random_fjsp_generator(Dataset):
    def __init__(
        self,
        num_sample,
        num_job,
        num_machine,
        time_low,
        time_high,
        device,
        **kwargs
    ):
        self.num_sample = num_sample
        self.num_machine = num_machine
        self.num_job = num_job
        self.time_low = time_low
        self.time_high = time_high
        self.device = device

        self.time_dev = 0.2
        self.opes_per_job_min = kwargs.get('opes_per_job_min',int(num_machine * 0.8))  # The minimum number of operations for a job
        self.opes_per_job_max = kwargs.get('opes_per_job_max',int(num_machine * 1.2))  # min==max is allowed

        self.mas_per_ope_min = 1  # The minimum number of machines that can process an operation
        self.mas_per_ope_max = num_machine
    
    def __len__(self):
        return self.num_sample

    def __getitem__(self, idx):

        self.nums_ope = [random.randint(self.opes_per_job_min, self.opes_per_job_max) for _ in range(self.num_job)]
        self.num_opes = sum(self.nums_ope)
        self.nums_option = [random.randint(self.mas_per_ope_min, self.mas_per_ope_max) for _ in
                            range(self.num_opes)]  # 每道工序可以由多少个machine加工
        self.num_options = sum(self.nums_option)
        self.ope_ma = []
        for val in self.nums_option:
            self.ope_ma = self.ope_ma + sorted(random.sample(range(1, self.num_machine + 1), val))  # 生成工序和machine的对应序列
        self.proc_time = []
        self.proc_times_mean = [random.randint(self.time_low, self.time_high) for _ in
                                range(self.num_opes)]
        for i in range(len(self.nums_option)):
            low_bound = max(self.time_low, round(self.proc_times_mean[i] * (1 - self.time_dev)))
            high_bound = min(self.time_high, round(self.proc_times_mean[i] * (1 + self.time_dev)))
            proc_time_ope = [random.randint(low_bound, high_bound) for _ in range(self.nums_option[i])]
            self.proc_time = self.proc_time + proc_time_ope
        self.num_ope_bias = [sum(self.nums_ope[0:i]) for i in range(self.num_job)]
        self.num_ma_bias = [sum(self.nums_option[0:i]) for i in range(self.num_opes)]

        fjsp_instance={
            'num_job': self.num_job,
            'num_machine': self.num_machine,
            'nums_ope': self.nums_ope, #每个job需要几道工序
            'proc_time': self.proc_time, #工序在不同机器上的加工时间
            'ope_ma': self.ope_ma,  # 工序可以在哪些机器上加工
            'num_ope_bias': self.num_ope_bias,
            'num_ma_bias': self.num_ma_bias,
        }

        return fjsp_instance





class customized_fjsp_loader(Dataset):
    def __init__(self, num_sample, device="cpu", path=None):
        self.num_sample = num_sample
        self.device = device
        self.path = path
        self.file_type = os.path.splitext(self.path)[-1]

        if self.file_type == ".fjs":
            #.fjs 一个文件一个instance
            flag=0
            datas = []
            with open(self.path) as file_object:
                lines = file_object.readlines()
                # datas.append(lines)
                processed_lines = []
                for line in lines:
                    if flag==0:
                        flag+=1
                        continue
                    line = line.strip()
                    str_nums = line.split()
                    nums = [float(x) if '.' in x else int(x) for x in str_nums]
                    processed_lines.append(nums)
                tensor = [torch.tensor(row, dtype=torch.int32) for row in processed_lines]
                padded_tensor = pad_sequence(tensor, batch_first=True, padding_value=-1)
            self.data = padded_tensor.unsqueeze(0).to(self.device)
            # self.data = datas

        else:
            raise NotImplementedError(
                f"The file type ({self.file_type}) is not supported in current version"
            )

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_sample



class public_fjsp_loader(Dataset):
    def __init__(self, device="cpu", path=None, **kwargs):
        self.device = device
        self.path = path
        self.data = {}
        self.num_sample = 0
        import os
        for root, dirs, files in os.walk(self.path):
            files.sort()
            for file in files:
                if file.endswith(".fjs"):
                    test_file = os.path.join(root, file)
                    with open(test_file) as file_object:
                        name = file.split(".")[0]
                        fjsp_instance = file_object.readlines()
                        num_job, num_machine, num_ope = nums_detec(fjsp_instance)
                        processed_lines = []
                        flag=0
                        for line in fjsp_instance:
                            if flag == 0:
                                flag += 1
                                continue
                            if line == '\n':
                                continue
                            line = line.strip()
                            str_nums = line.split()
                            nums = [float(x) if '.' in x else int(x) for x in str_nums]
                            processed_lines.append(nums)
                        tensor = [torch.tensor(row, dtype=torch.int32) for row in processed_lines]
                        padded_tensor = pad_sequence(tensor, batch_first=True, padding_value=-1)
                        self.data[name] = {"num_job":num_job,"num_machine":num_machine,"num_ope":num_ope,"instance":padded_tensor}
                        self.num_sample += 1

        if self.num_sample == 0:
            raise ValueError(f"No FJS files found in {self.path}")
    def __getitem__(self, index):
        name = list(self.data.keys())[index]
        data = self.data[name]
        if "Mk" in name:
            optimal = brandimarte_cost.get(name,0)
        elif "e_la" in name:
            optimal = e_la_cost.get(name,0)
        elif "r_la" in name:
            optimal = r_la_cost.get(name,0)
        elif "v_la" in name:
            optimal = v_la_cost.get(name,0)

        return {
            "name": name,
            "data": data["instance"],
            "num_job":data["num_job"],
            "num_machine": data["num_machine"],
            "num_ope": data["num_ope"],
            "optimal": optimal,
        }
    def __len__(self):
        return self.num_sample