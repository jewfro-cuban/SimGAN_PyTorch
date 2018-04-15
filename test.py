import os
import time
import torch
import torch.utils.data as Data
import torchvision
from torch.autograd import Variable
from torchvision import transforms
from torch import nn
import numpy as np
from utils.image_history_buffer import ImageHistoryBuffer
from utils.network import Discriminator, Refiner
from utils.external_func import get_accuracy, loop_iter, MyTimer
import config as cfg
from PIL import Image


class Main(object):
    def __init__(self):
        # network
        self.G = None
        self.fake_images_loader = None

    def build_network(self):
        print('Building network ...')
        self.G = Refiner(4, cfg.img_channels, nb_features=64)

        if cfg.cuda_use:
            self.G.cuda(cfg.cuda_num)

    def load_previous_mode(self):
        if not os.path.isdir(cfg.save_path):
            os.mkdir(cfg.save_path)
        ckpts = os.listdir(cfg.save_path)
        refiner_ckpts = [ckpt for ckpt in ckpts if 'R_' in ckpt]
        refiner_ckpts.sort(key=lambda x: int(x[2:-4]), reverse=True)

        assert len(refiner_ckpts) > 0, 'Cannot find any save model'
        # Load pretrained model
        print('Loading previous model ...'.format(refiner_ckpts[0]))
        self.G.load_state_dict(torch.load(os.path.join(cfg.save_path, refiner_ckpts[0])))


    def get_data_loaders(self):
        print('Creating dataloaders ...')
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        fake_folder = torchvision.datasets.ImageFolder(root='data/Eval', transform=transform)

        self.fake_images_loader = Data.DataLoader(fake_folder, batch_size=1, shuffle=False,
                                                  pin_memory=True, drop_last=False, num_workers=3)

    def eval(self):
        print('Start Evaluating ...')
        transform_mean = 0.
        transform_std = 1.
        for trans in self.fake_images_loader.dataset.transform.transforms:
            if isinstance(trans, torchvision.transforms.transforms.Normalize):
                transform_mean = trans.mean
                transform_std = trans.std

        self.G.eval()
        for i, (image, _) in enumerate(self.fake_images_loader):
            fake_images = Variable(image).cuda(cfg.cuda_num)
            refined_images = self.G(fake_images)
            refined_images = refined_images.cpu().data.numpy()
            if refined_images.shape[1] == 1:
                refined_images = np.squeeze(refined_images, 0)
                mean = [transform_mean[0]]
                std = [transform_std[0]]
            mean = np.expand_dims(np.expand_dims(np.array(mean), 1), 2)
            std = np.expand_dims(np.expand_dims(np.array(std), 1), 2)
            refined_images = ((refined_images * std + mean) * 256).astype(np.uint8)
            refined_images = Image.fromarray(refined_images.squeeze(0))
            filename = self.fake_images_loader.dataset.imgs[i][0].split('/')[-1]
            refined_images.save(os.path.join('data/Eval/Results', filename))



if __name__ == '__main__':
    obj = Main()
    obj.build_network()
    obj.get_data_loaders()
    obj.load_previous_mode()
    obj.eval()



