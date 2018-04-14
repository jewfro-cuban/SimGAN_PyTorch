import torch
from torch.autograd import Variable
import config as cfg
import torch.nn.functional as F
import time


def get_accuracy(output, type='real'):
    assert type in ['real', 'refine']

    if type == 'real':
        label = Variable(torch.zeros(output.size(0)).type(torch.LongTensor)).cuda(cfg.cuda_num)
    else:
        label = Variable(torch.ones(output.size(0)).type(torch.LongTensor)).cuda(cfg.cuda_num)

    softmax_output = F.softmax(output, dim=-1)
    acc = softmax_output.data.max(1)[1].cpu().numpy() == label.data.cpu().numpy()
    return acc.mean()


def loop_iter(dataloader):
    while True:
        for data in iter(dataloader):
            yield data


class MyTimer():
    def __init__(self):
        self.time_dict = {}
        self.count_dict = {}
        self.t0 = 0.

    def track(self):
        self.t0 = time.time()

    def add_value(self, title):
        value = time.time() - self.t0
        self.t0 = 0.

        if not title in self.time_dict:
            self.time_dict[title] = value
            self.count_dict[title] = 1
        else:
            self.time_dict[title] = (self.time_dict[title] * self.count_dict[title] + value)
            self.count_dict[title] += 1
            self.time_dict[title] /= self.count_dict[title]

    def get_all_time(self):
        return self.time_dict