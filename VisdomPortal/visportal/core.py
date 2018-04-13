import visdom
import numpy as np
import torch


class VisdomPortal():
    def __init__(self, env_name='Portal_Default', image_norm=[(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]):
        # prepare visdom
        self.vis = visdom.Visdom(env=env_name)
        self.win_handles = {}
        self.image_norm = image_norm[0]
        self.image_std = image_norm[1]

    def draw_curve(self, value, step, title):
        assert isinstance(value, (torch.autograd.variable.Variable, int, float,
                                  np.ndarray)), 'Value type should be torch variable, int, float or numpy.ndarray.'
        assert isinstance(step, (int, float, np.ndarray)), 'Step type should be int, float or numpy.ndarray.'

        if isinstance(value, torch.autograd.variable.Variable):
            if value.is_cuda:
                value = value.cpu()
            value = value.data.numpy().astype(np.float)
        elif isinstance(value, (int, float)):
            value = np.array([value], dtype=np.float)

        if isinstance(step, np.ndarray):
            step = step.astype(int)
        else:
            step = np.array([step], dtype=int)

        if title in self.win_handles:
            win_name = self.win_handles[title]
            self.vis.line(value, step, win=win_name, update='append', opts={'title': title})
        else:
            self.win_handles[title] = self.vis.line(value, step, opts={'title': title})

    def draw_images(self, value, title, unnormalize=True):
        assert isinstance(value, (torch.autograd.variable.Variable, torch._TensorBase,
                                  np.ndarray)), 'Value type should be torch variable, torch tensor or numpy.ndarray.'
        # variable
        if isinstance(value, torch.autograd.variable.Variable):
            if value.is_cuda:
                value = value.cpu()
            value = value.data.numpy()
        # tensor
        elif isinstance(value, torch._TensorBase):
            if value.is_cuda:
                value = value.cpu()
            value = value.numpy()

        value = value.astype(np.float)
        assert len(value.shape) == 4, 'value must have 4 dimensions'
        if unnormalize:
            norm = np.array(self.image_norm[:value.shape[1]]).reshape((1, -1, 1, 1))
            std = np.array(self.image_std[:value.shape[1]]).reshape((1, -1, 1, 1))
            value = (value * std + norm) * 256

        if title in self.win_handles:
            win_name = self.win_handles[title]
            self.vis.images(value.astype(np.uint8), win=win_name,
                            opts={'caption': title})
        else:
            self.win_handles[title] = self.vis.images(value.astype(np.uint8),
                                                      opts={'caption': title})
