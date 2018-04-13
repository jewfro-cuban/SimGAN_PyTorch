import torch
import torch.utils.data as Data
import torchvision
from VisdomPortal.visportal.core import VisdomPortal
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from lib.image_history_buffer import ImageHistoryBuffer
from lib.network import Discriminator, Refiner
from lib.image_utils import generate_img_batch, calc_acc
import config as cfg
import os

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

        # data
        self.syn_train_loader = None
        self.real_loader = None
        self.current_step = 0

    def build_network(self):
        print('Building network ...')
        self.G = Refiner(4, cfg.img_channels, nb_features=64)
        self.D = Discriminator(input_features=cfg.img_channels)

        if cfg.cuda_use:
            self.G.cuda(cfg.cuda_num)
            self.D.cuda(cfg.cuda_num)

        self.refiner_optimizer = torch.optim.SGD(self.G.parameters(), lr=cfg.r_lr)
        self.discriminator_loss = torch.optim.SGD(self.D.parameters(), lr=cfg.d_lr)
        self.combined_optimizer = torch.optim.SGD([
            {'params': self.G.parameters()},
            {'params': self.D.parameters()}
        ], lr=cfg.d_lr)

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

    def load_data(self):
        print('Creating dataloaders ...')
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Scale((cfg.img_width, cfg.img_height)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        syn_train_folder = torchvision.datasets.ImageFolder(root=cfg.syn_path, transform=transform)
        self.syn_train_loader = Data.DataLoader(syn_train_folder, batch_size=cfg.batch_size, shuffle=True,
                                                pin_memory=True)
        print('syn_train_batch %d' % len(self.syn_train_loader))
        real_folder = torchvision.datasets.ImageFolder(root=cfg.real_path, transform=transform)
        self.real_loader = Data.DataLoader(real_folder, batch_size=cfg.batch_size, shuffle=True,
                                           pin_memory=True)
        print('real_batch %d' % len(self.real_loader))

    def pretrain_generator(self):
        # we first train the Rθ network with just self-regularization loss for 1,000 steps
        print('pre-training the refiner network %d times...' % cfg.r_pretrain)
        self.G.train()
        for index in range(cfg.r_pretrain):
            faked_images, _ = self.syn_train_loader.__iter__().next()
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
            if (index % cfg.r_pre_per == 0) or (index == cfg.r_pretrain - 1):
                print('[%d/%d] (R)reg_loss: %.4f' % (index, cfg.r_pretrain, reg_loss.data[0]))
                torch.save(self.G.state_dict(), os.path.join(cfg.save_path, 'R_0.pkl'))

    def pretrain_discrimintor(self):
        # and Dφ for 200 steps (one mini-batch for refined images, another for real)
        print('pre-training the discriminator network %d times...' % cfg.r_pretrain)
        self.D.train()
        self.G.eval()
        for index in range(cfg.d_pretrain):
            real_images, _ = self.real_loader.__iter__().next()
            real_images = Variable(real_images).cuda(cfg.cuda_num)
            fake_images, _ = self.syn_train_loader.__iter__().next()
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
            acc_real = calc_acc(real_predictions, 'real')
            acc_ref = calc_acc(refined_predictions, 'refine')
            d_loss_real = self.local_adversarial_loss(real_predictions, real_labels)
            d_loss_refn = self.local_adversarial_loss(refined_predictions, refined_labels)
            d_loss = d_loss_real + d_loss_refn

            self.discriminator_loss.zero_grad()
            d_loss.backward()
            self.discriminator_loss.step()

            if index % cfg.d_pre_per == 0 or (index == cfg.d_pretrain - 1):
                print('[%d/%d] (D)d_loss:%f  acc_real:%.2f%% acc_ref:%.2f%%'
                      % (index, cfg.d_pretrain, d_loss.data[0], acc_real, acc_ref))

        print('Save D_pre to models/D_0.pkl')
        torch.save(self.D.state_dict(), os.path.join(cfg.save_path, 'D_0.pkl'))

    def train(self):
        print('Start Formal Training')

        # image_history_buffer = ImageHistoryBuffer((0, cfg.img_channels, cfg.img_height, cfg.img_width),
        #                                           cfg.buffer_size * 10, cfg.batch_size)

        assert self.current_step < cfg.train_steps, 'Target step is smaller than current step!'

        for step in range(self.current_step, cfg.train_steps):
            print('Step[%d/%d]' % (step, cfg.train_steps))
            self.current_step = step
            self.D.train()
            self.G.train()

            for index in range(cfg.k_g*2):
                fake_images, _ = self.syn_train_loader.__iter__().next()
                # forward
                fake_images = Variable(fake_images).cuda(cfg.cuda_num)
                refined_images = self.G(fake_images)
                refined_predictions = self.D(refined_images).view(-1, 2)
                refined_labels = Variable(torch.zeros(refined_predictions.size(0)).type(torch.LongTensor)).cuda(cfg.cuda_num)
                # calculate loss
                reg_loss = self.self_regularization_loss(refined_images, fake_images)
                reg_loss = torch.mul(reg_loss, self.delta)
                adv_loss = self.local_adversarial_loss(refined_predictions, refined_labels)
                r_loss = reg_loss + adv_loss
                # update
                self.combined_optimizer.zero_grad()
                r_loss.backward()
                self.combined_optimizer.step()

            print('(R)r_loss:%.4f r_loss_reg:%.4f, r_loss_adv:%.4f' % (r_loss.data[0], reg_loss.data[0], adv_loss.data[0]))

            # ========= train the D =========
            self.G.eval()
            self.D.train()

            for index in range(cfg.k_d):
                # generate refined images
                fake_images, _ = self.syn_train_loader.__iter__().next()
                fake_images = Variable(fake_images).cuda(cfg.cuda_num)
                refined_images = self.G(fake_images)
                refined_images = refined_images.detach()
                # real images
                real_images, _ = self.real_loader.__iter__().next()
                real_images = Variable(real_images).cuda(cfg.cuda_num)
                # visualization
                vis.draw_images(real_images, 'Real Images')
                vis.draw_images(fake_images, 'Simulated Images')
                vis.draw_images(refined_images, 'Refined Images')

                # use a history of refined images
                # half_batch_from_image_history = image_history_buffer.get_from_image_history_buffer()
                # image_history_buffer.add_to_image_history_buffer(ref_image_batch.cpu().data.numpy())
                #
                # if len(half_batch_from_image_history):
                #     torch_type = torch.from_numpy(half_batch_from_image_history)
                #     v_type = Variable(torch_type).cuda(cfg.cuda_num)
                #     ref_image_batch[:cfg.batch_size // 2] = v_type


                # predict real images
                real_predictions = self.D(real_images).view(-1, 2)
                real_labels = Variable(torch.zeros(real_predictions.size(0)).type(torch.LongTensor)).cuda(
                    cfg.cuda_num)
                pred_loss_real = self.local_adversarial_loss(real_predictions, real_labels)
                acc_real = calc_acc(real_predictions, 'real')
                # predict refined images
                refined_predictions = self.D(refined_images).view(-1, 2)
                refined_labels = Variable(torch.ones(refined_predictions.size(0)).type(torch.LongTensor)).cuda(
                    cfg.cuda_num)
                pred_loss_refn = self.local_adversarial_loss(refined_predictions, refined_labels)
                acc_ref = calc_acc(refined_predictions, 'refine')
                d_loss = pred_loss_refn + pred_loss_real

                self.D.zero_grad()
                d_loss.backward()
                self.discriminator_loss.step()

                print('(D)d_loss:%.4f real_loss:%.4f(%.2f%%) refine_loss:%.4f(%.2f%%)'
                      % (d_loss.data[0] / 2, pred_loss_real.data[0], acc_real, pred_loss_refn.data[0], acc_ref))

            if step % 5 == 0:
                vis.draw_curve(value=r_loss, step=step, title='Refiner Loss')
                vis.draw_curve(value=d_loss, step=step, title='Discriminator Loss')
                vis.draw_curve(value=acc_real, step=step, title='Real Images Discriminator Accuracy')
                vis.draw_curve(value=acc_ref, step=step, title='Refined Images Discriminator Accuracy')

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
    obj.load_data()

    if obj.load_previous_mode():
        obj.pretrain_generator()
        obj.pretrain_discrimintor()

    obj.train()

    obj.generate_all_train_image()


