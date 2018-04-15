import os
import time
import torch
import torch.utils.data as Data
import torchvision
from torch.autograd import Variable
from torchvision import transforms
from torch import nn

from VisdomPortal.visportal.core import VisdomPortal
from utils.image_history_buffer import ImageHistoryBuffer
from utils.network import Discriminator, Refiner
from utils.external_func import get_accuracy, loop_iter, MyTimer
import config as cfg


vis = VisdomPortal(env_name='SimGAN_{}'.format('Eye'))


class Main(object):
    def __init__(self):
        # network
        self.G = None
        self.D = None
        self.refiner_optimizer = None
        self.discriminator_loss = None
        self.combined_optimizer = None
        self.self_regularization_loss = None
        self.local_adversarial_loss = None
        self.delta = None
        self.my_timer = MyTimer()
        # data
        self.fake_images_loader = None
        self.real_images_loader = None
        self.fake_images_iter = None
        self.real_images_iter = None
        self.current_step = 0

    def build_network(self):
        print('Building network ...')
        self.G = Refiner(4, cfg.img_channels, nb_features=64)
        self.D = Discriminator(input_features=cfg.img_channels)

        if cfg.cuda_use:
            self.G.cuda(cfg.cuda_num)
            self.D.cuda(cfg.cuda_num)

        self.refiner_optimizer = torch.optim.Adam(self.G.parameters(), lr=cfg.init_lr)
        self.discriminator_loss = torch.optim.SGD(self.D.parameters(), lr=cfg.init_lr)
        self.combined_optimizer = torch.optim.SGD([
            {'params': self.G.parameters()},
            {'params': self.D.parameters()}
        ], lr=cfg.init_lr)

        self.self_regularization_loss = nn.L1Loss(size_average=False)
        self.local_adversarial_loss = nn.CrossEntropyLoss(size_average=True)
        self.delta = cfg.delta

    def load_previous_mode(self):
        if not os.path.isdir(cfg.save_path):
            os.mkdir(cfg.save_path)
        ckpts = os.listdir(cfg.save_path)
        refiner_ckpts = [ckpt for ckpt in ckpts if 'R_' in ckpt]
        refiner_ckpts.sort(key=lambda x: int(x[2:-4]), reverse=True)
        discriminator_ckpts = [ckpt for ckpt in ckpts if 'D_' in ckpt]
        discriminator_ckpts.sort(key=lambda x: int(x[2:-4]), reverse=True)

        if len(refiner_ckpts) == 0 or len(discriminator_ckpts) == 0 or not os.path.isfile(os.path.join(cfg.save_path, cfg.optimizer_path)):
            return True

        optimizer_status = torch.load(os.path.join(cfg.save_path, cfg.optimizer_path))
        self.refiner_optimizer.load_state_dict(optimizer_status['optR'])
        self.discriminator_loss.load_state_dict(optimizer_status['optD'])
        self.combined_optimizer.load_state_dict(optimizer_status['optC'])
        self.current_step = optimizer_status['step']

        # Load pretrained model
        print('Loading previous model {} and {} ...'.format(refiner_ckpts[0], discriminator_ckpts[0]))
        self.D.load_state_dict(torch.load(os.path.join(cfg.save_path, discriminator_ckpts[0])))
        self.G.load_state_dict(torch.load(os.path.join(cfg.save_path, refiner_ckpts[0])))

        return False

    def get_data_loaders(self):
        print('Creating dataloaders ...')
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((cfg.img_width, cfg.img_height),  interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        fake_folder = torchvision.datasets.ImageFolder(root=cfg.syn_path, transform=transform)

        self.fake_images_loader = Data.DataLoader(fake_folder, batch_size=cfg.batch_size, shuffle=True,
                                                  pin_memory=False, drop_last=True, num_workers=3)

        real_folder = torchvision.datasets.ImageFolder(root=cfg.real_path, transform=transform)
        self.real_images_loader = Data.DataLoader(real_folder, batch_size=cfg.batch_size, shuffle=True,
                                                  pin_memory=False, drop_last=True, num_workers=3)
        self.fake_images_iter = loop_iter(self.fake_images_loader)
        self.real_images_iter = loop_iter(self.real_images_loader)


    def pretrain_generator(self):
        # we first train the Rθ network with just self-regularization loss for 1,000 steps
        print('pre-training the refiner network %d times...' % cfg.g_pretrain)
        self.G.train()
        # fake_iter = iter(self.fake_images_loader)
        for step in range(cfg.g_pretrain):
            faked_images, _ = next(self.fake_images_iter)
            faked_images = Variable(faked_images).cuda(cfg.cuda_num)
            refined_images = self.G(faked_images)
            # regularization loss
            reg_loss = self.self_regularization_loss(refined_images, faked_images)
            reg_loss = torch.mul(reg_loss, self.delta)
            # update model
            self.refiner_optimizer.zero_grad()
            reg_loss.backward()
            self.refiner_optimizer.step()
            # save
            if (step % cfg.r_pre_per == 0) or (step == cfg.g_pretrain - 1):
                print('------Step[%d/%d]------' % (step, cfg.g_pretrain))
                print('# Refiner: loss: %.4f' % (reg_loss.data[0]))
                vis.draw_curve(value=reg_loss, step=step, title='Pretrain Refiner Loss')
                torch.save(self.G.state_dict(), os.path.join(cfg.save_path, 'R_0.pkl'))

    def pretrain_discrimintor(self):
        # and Dφ for 200 steps (one mini-batch for refined images, another for real)
        print('pre-training the discriminator network %d times...' % cfg.d_pretrain)
        self.D.train()
        self.G.eval()
        for step in range(cfg.d_pretrain):
            real_images, _ = next(self.real_images_iter)
            # real_images, _ = self.real_images_loader.__iter__().next()
            real_images = Variable(real_images).cuda(cfg.cuda_num)
            fake_images, _ = next(self.fake_images_iter)
            # fake_images, _ = self.fake_images_loader.__iter__().next()
            fake_images = Variable(fake_images).cuda(cfg.cuda_num)
            # refine fake images
            refined_images = self.G(fake_images)
            # prediction
            real_predictions = self.D(real_images).view(-1, 2)
            refined_predictions = self.D(refined_images).view(-1, 2)
            # create labels
            real_labels = Variable(torch.zeros(real_predictions.size(0)).type(torch.LongTensor)).cuda(cfg.cuda_num)
            refined_labels = Variable(torch.ones(real_predictions.size(0)).type(torch.LongTensor)).cuda(cfg.cuda_num)
            # loss and accuracy
            acc_real = get_accuracy(real_predictions, 'real')
            acc_ref = get_accuracy(refined_predictions, 'refine')
            d_loss_real = self.local_adversarial_loss(real_predictions, real_labels)
            d_loss_refn = self.local_adversarial_loss(refined_predictions, refined_labels)
            d_loss = d_loss_real + d_loss_refn

            self.discriminator_loss.zero_grad()
            d_loss.backward()
            self.discriminator_loss.step()

            if step % cfg.d_pre_per == 0 or (step == cfg.d_pretrain - 1):
                vis.draw_curve(value=d_loss, step=step, title='Pretrain Discriminator Loss')
                print('------Step[%d/%d]------' % (step, cfg.d_pretrain))
                print('# Discriminator: loss:%f  accuracy_real:%.2f accuracy_ref:%.2f'
                      % (d_loss.data[0], acc_real, acc_ref))

        print('Save D_pre to models/D_0.pkl')
        torch.save(self.D.state_dict(), os.path.join(cfg.save_path, 'D_0.pkl'))

    def train(self):
        print('Start Formal Training ...')

        image_history_buffer = ImageHistoryBuffer((0, cfg.img_channels, cfg.img_height, cfg.img_width),
                                                  cfg.buffer_size, cfg.batch_size)

        assert self.current_step < cfg.train_steps, 'Target step is smaller than current step!'
        step_timer = time.time()
        for step in range(self.current_step, cfg.train_steps):

            self.current_step = step
            self.D.train()
            self.G.train()

            for index in range(cfg.k_g*2):
                self.my_timer.track()
                fake_images, _ = next(self.fake_images_iter)
                fake_images = Variable(fake_images).cuda(cfg.cuda_num)
                self.my_timer.add_value('Read Fake Images')
                # forward #1
                self.my_timer.track()
                refined_images = self.G(fake_images)
                self.my_timer.add_value('Refine Fake Images')
                # forward #2
                self.my_timer.track()
                refined_predictions = self.D(refined_images).view(-1, 2)
                self.my_timer.add_value('Predict Fake Images')
                # calculate loss
                self.my_timer.track()
                refined_labels = Variable(torch.zeros(refined_predictions.size(0)).type(torch.LongTensor)).cuda(cfg.cuda_num)
                reg_loss = self.self_regularization_loss(refined_images, fake_images)
                reg_loss = torch.mul(reg_loss, self.delta)
                adv_loss = self.local_adversarial_loss(refined_predictions, refined_labels)
                r_loss = reg_loss + adv_loss
                self.my_timer.add_value('Get Refine Loss')
                # backward
                self.my_timer.track()
                self.combined_optimizer.zero_grad()
                r_loss.backward()
                self.combined_optimizer.step()
                self.my_timer.add_value('Backward Refine Loss')

            # ========= train the D =========
            self.G.eval()
            self.D.train()

            for index in range(cfg.k_d):
                # get images
                self.my_timer.track()
                fake_images, _ = next(self.fake_images_iter)
                fake_images = Variable(fake_images).cuda(cfg.cuda_num)
                real_images, _ = next(self.real_images_iter)
                real_images = Variable(real_images).cuda(cfg.cuda_num)
                self.my_timer.add_value('Read All Images')
                # generate refined images
                self.my_timer.track()
                refined_images = self.G(fake_images)
                self.my_timer.add_value('Refine Fake Images')
                # use a history of refined images
                self.my_timer.track()
                refined_images = refined_images.detach()
                half_batch_from_image_history = image_history_buffer.get_from_image_history_buffer()
                image_history_buffer.add_to_image_history_buffer(refined_images.cpu().data.numpy())
                if len(half_batch_from_image_history):
                    history_refined_images = torch.from_numpy(half_batch_from_image_history)
                    history_refined_images = Variable(history_refined_images).cuda(cfg.cuda_num)
                    refined_images[:cfg.batch_size // 2] = history_refined_images
                self.my_timer.add_value('Get History Images')
                # predict images
                self.my_timer.track()
                real_predictions = self.D(real_images).view(-1, 2)
                refined_predictions = self.D(refined_images).view(-1, 2)
                self.my_timer.add_value('Predict All Images')
                # get all loss
                self.my_timer.track()
                real_labels = Variable(torch.zeros(real_predictions.size(0)).type(torch.LongTensor)).cuda(
                    cfg.cuda_num)
                pred_loss_real = self.local_adversarial_loss(real_predictions, real_labels)
                acc_real = get_accuracy(real_predictions, 'real')
                refined_labels = Variable(torch.ones(refined_predictions.size(0)).type(torch.LongTensor)).cuda(
                    cfg.cuda_num)
                pred_loss_refn = self.local_adversarial_loss(refined_predictions, refined_labels)
                acc_ref = get_accuracy(refined_predictions, 'refine')
                d_loss = pred_loss_refn + pred_loss_real
                self.my_timer.add_value('Get Combine Loss')
                self.my_timer.track()
                self.D.zero_grad()
                d_loss.backward()
                self.discriminator_loss.step()
                self.my_timer.add_value('Backward Combine Loss')

            if step % cfg.d_pre_per == 0:
                print('------Step[%d/%d]------Time Cost: %.2f seconds' % (step, cfg.train_steps, time.time() - step_timer))
                print('# Refiner: loss:%.4f reg_loss:%.4f, adv_loss:%.4f' % (
                    r_loss.data[0], reg_loss.data[0], adv_loss.data[0]))
                print('# Discrimintor: loss:%.4f real:%.4f(%.2f) refined:%.4f(%.2f)'
                      % (d_loss.data[0] / 2, pred_loss_real.data[0], acc_real, pred_loss_refn.data[0], acc_ref))

                # visualization
                vis.draw_images(real_images, 'Real Images')
                vis.draw_images(fake_images, 'Simulated Images')
                vis.draw_images(refined_images, 'Refined Images')
                vis.draw_curve(value=r_loss, step=step, title='Refiner Loss')
                vis.draw_curve(value=d_loss, step=step, title='Discriminator Loss')
                vis.draw_curve(value=acc_real, step=step, title='Real Images Discriminator Accuracy')
                vis.draw_curve(value=acc_ref, step=step, title='Refined Images Discriminator Accuracy')

                time_dict = self.my_timer.get_all_time()
                vis.draw_bars(time_dict, 'Time Cost (second)')
                step_timer = time.time()

            if step % cfg.save_per == 0 and step > 0:
                print('Save two model dict.')
                torch.save(self.D.state_dict(), os.path.join(cfg.save_path, cfg.D_path % step))
                torch.save(self.G.state_dict(), os.path.join(cfg.save_path, cfg.R_path % step))
                state = {
                    'step': step,
                    'optD': self.discriminator_loss.state_dict(),
                    'optR': self.refiner_optimizer.state_dict(),
                    'optC': self.combined_optimizer.state_dict(),
                }
                torch.save(state, os.path.join(cfg.save_path, cfg.optimizer_path))



if __name__ == '__main__':
    obj = Main()
    obj.build_network()
    obj.get_data_loaders()

    if obj.load_previous_mode():
        obj.pretrain_generator()
        obj.pretrain_discrimintor()

    obj.train()

    obj.generate_all_train_image()


