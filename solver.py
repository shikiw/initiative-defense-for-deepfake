from model import Attackmodel
from model import PerturbationDiscriminator
from model import Generator
from model import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime

from PIL import Image
from torchvision import transforms as T
# import defenses.smoothing as smoothing
from networks import define_G
from model import AutoEncoder
from HidingRes import HidingRes

class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, celeba_loader, rafd_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.celeba_loader = celeba_loader
        self.rafd_loader = rafd_loader

        # Model configurations.
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs

        # Test configurations.
        self.use_PG = config.use_PG
        self.test_iters = config.test_iters

        # Attack-Training configurations.
        self.eps = config.eps
        self.atk_lr = config.atk_lr
        self.model_iters = config.model_iters
        self.attack_iters = config.attack_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        if self.dataset in ['CelebA', 'RaFD']:
            self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)
            self.lastiter_G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
            self.lastiter_D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)
        elif self.dataset in ['Both']:
            self.G = Generator(self.g_conv_dim, self.c_dim+self.c2_dim+2, self.g_repeat_num)   # 2 for mask vector.
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim+self.c2_dim, self.d_repeat_num)

        self.G = torch.nn.DataParallel(self.G)
        self.D = torch.nn.DataParallel(self.D)
        self.lastiter_G = torch.nn.DataParallel(self.lastiter_G)
        self.lastiter_D = torch.nn.DataParallel(self.lastiter_D)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])

        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
            
        self.G.to(self.device)
        self.D.to(self.device)
        self.lastiter_G.to(self.device)
        self.lastiter_D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def restore_clean_model(self, model_iters):
        """Restore the trained generator and discriminator for correspondence."""
        print('Loading the trained models from step {}...'.format(model_iters))
        if self.dataset in ['CelebA', 'RaFD']:
            self.clean_G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
            # self.clean_G = define_G(3 + self.c_dim, 3, self.g_conv_dim, 'unet_128', 'instance', not True, 'normal', 0.02)
            # self.clean_G = AutoEncoder(self.g_conv_dim, self.c_dim)
            # self.clean_G = HidingRes(in_c=3 + self.c_dim)
            # self.clean_D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)

        self.clean_G = torch.nn.DataParallel(self.clean_G)
        self.clean_G.to(self.device)
        G_path = os.path.join('checkpoints/', '{}-G.ckpt'.format(model_iters))
        self.clean_G.module.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))


    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr, atk_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr
        for param_group in self.pg_optimizer.param_groups:
            param_group['lr'] = atk_lr

    def clear_grad(self, model):
        """Clear gradient buffers of model."""
        for p in model.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
    
    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        if dataset == 'CelebA':
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'CelebA':
                c_trg = c_org.clone()
                if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
            elif dataset == 'RaFD':
                c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def classification_loss(self, logit, target, dataset='CelebA'):
        """Compute binary or softmax cross entropy loss."""
        if dataset == 'CelebA':
            return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
        elif dataset == 'RaFD':
            return F.cross_entropy(logit, target)

    def train(self):
        """Train StarGAN within a single dataset."""
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_fixed, c_org = next(data_iter)
        x_fixed = x_fixed.to(self.device)
        c_fixed_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, label_org = next(data_iter)

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            if self.dataset == 'CelebA':
                c_org = label_org.clone()
                c_trg = label_trg.clone()
            elif self.dataset == 'RaFD':
                c_org = self.label2onehot(label_org, self.c_dim)
                c_trg = self.label2onehot(label_trg, self.c_dim)
            else:
                exit(0)

            x_real = x_real.to(self.device)           # Input images.
            c_org = c_org.to(self.device)             # Original domain labels.
            c_trg = c_trg.to(self.device)             # Target domain labels.
            label_org = label_org.to(self.device)     # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)     # Labels for computing classification loss.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls = self.D(x_real)
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label_org, self.dataset)

            # Compute loss with fake images.
            x_fake = self.G(x_real, c_trg)
            out_src, out_cls = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()
            
            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            
            if (i+1) % self.n_critic == 0:
                # Original-to-target domain.
                x_fake = self.G(x_real, c_trg)
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg, self.dataset)

                # Target-to-original domain.
                x_reconst = self.G(x_fake, c_org)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_fixed_list:
                        x_fake_list.append(self.G(x_fixed, c_fixed))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))


    def train_PG(self, d_lr, g_lr, atk_lr, data_loader, data_iter, x_fixed, c_fixed_list, start_time, i):

        # =================================================================================== #
        #                             1. Preprocess input data                                #
        # =================================================================================== #
        
        # Fetch real images and labels.
        try:
            x_real, label_org = next(data_iter)
        except:
            data_iter = iter(data_loader)
            x_real, label_org = next(data_iter)
        
        # Generate target domain labels randomly.
        rand_idx = torch.randperm(label_org.size(0))
        label_trg = label_org[rand_idx]
        
        if self.dataset == 'CelebA':
            c_org = label_org.clone()
            c_trg = label_trg.clone()
        elif self.dataset == 'RaFD':
            c_org = self.label2onehot(label_org, self.c_dim)
            c_trg = self.label2onehot(label_trg, self.c_dim)
        
        x_real = x_real.to(self.device)           # Input images.
        c_org = c_org.to(self.device)             # Original domain labels.
        c_trg = c_trg.to(self.device)             # Target domain labels.
        label_org = label_org.to(self.device)     # Labels for computing classification loss.
        label_trg = label_trg.to(self.device)     # Labels for computing classification loss.

        if i > 0:
            self.lastiter_G.load_state_dict(self.G.state_dict())
            self.lastiter_D.load_state_dict(self.D.state_dict())

        # =================================================================================== #
        #                   2.1. Get paras of StarGAN in One-Step-Update                      #
        # =================================================================================== #

        self.PG.train()
        self.PD.train()
        
        # =================================================================================== #
        #                           2.2. Train the discriminator                              #
        # =================================================================================== #

        # Compute loss with real images.
        out_src, out_cls = self.D(x_real)
        d_loss_real = - torch.mean(out_src)
        d_loss_cls = self.classification_loss(out_cls, label_org, self.dataset)

        # Compute loss with fake images.
        x_fake = self.G(x_real, c_trg)
        out_src, out_cls = self.D(x_fake.detach())
        d_loss_fake = torch.mean(out_src)

        # Compute loss for gradient penalty.
        alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
        x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
        out_src, _ = self.D(x_hat)
        d_loss_gp = self.gradient_penalty(out_src, x_hat)

        # Backward and optimize.
        d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
        self.reset_grad()
        d_loss.backward()
        self.d_optimizer.step()

        # Logging.
        loss = {}
        loss['D/loss_real'] = d_loss_real.item()
        loss['D/loss_fake'] = d_loss_fake.item()
        loss['D/loss_cls'] = d_loss_cls.item()
        loss['D/loss_gp'] = d_loss_gp.item()


        # =================================================================================== #
        #                             2.3. Train the generator                                #
        # =================================================================================== #

        if (i + 1) % self.n_critic == 0:
            # Original-to-target domain.
            x_fake = self.G(x_real, c_trg)
            out_src, out_cls = self.D(x_fake)
            g_loss_fake = - torch.mean(out_src)
            g_loss_cls = self.classification_loss(out_cls, label_trg, self.dataset)

            # Target-to-original domain.
            x_reconst = self.G(x_fake, c_org)
            g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

            # Backward and optimize.
            g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
            self.reset_grad()
            g_loss.backward()
            self.g_optimizer.step()

            # Logging.
            loss['G/loss_fake'] = g_loss_fake.item()
            loss['G/loss_rec'] = g_loss_rec.item()
            loss['G/loss_cls'] = g_loss_cls.item()

        # =================================================================================== #
        #                       3. Update PD using current paras of PG.                       #
        # =================================================================================== #

        # Compute loss with no-perturbed images.
        output = self.PD(x_real)
        pd_loss_real = - torch.mean(output)

        # Compute loss with perturbed images.
        pert = self.PG(x_real) * self.eps
        x_adv = torch.clamp(x_real + pert, -1.0, 1.0)
        output = self.PD(x_adv.detach())
        pd_loss_fake = torch.mean(output)

        # Compute loss for gradient penalty.
        alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
        x_hat = (alpha * x_real.data + (1 - alpha) * x_adv.data).requires_grad_(True)
        output = self.PD(x_hat)
        pd_loss_gp = self.gradient_penalty(output, x_hat)

        # Backward and optimize.
        pd_loss = pd_loss_real + pd_loss_fake + self.lambda_gp * pd_loss_gp
        self.pd_optimizer.zero_grad()
        pd_loss.backward()
        self.pd_optimizer.step()

        # Logging.
        loss = {}
        loss['PD/loss_real'] = pd_loss_real.item()
        loss['PD/loss_fake'] = pd_loss_fake.item()
        loss['PD/loss_gp'] = pd_loss_gp.item()

        # =================================================================================== #
        #                4. Update attack model using current paras of StarGAN                #
        # =================================================================================== #

        # Get adversarial data with PG.
        pert = self.PG(x_real) * self.eps
        x_adv = torch.clamp(x_real + pert, -1.0, 1.0)

        # Get the traversal of target label list.
        c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

        # Maximum the transfer loss, the reconstruction loss and the Discriminator loss at current iteration.
        eta_sum = 0.0
        pg_loss_compare = 0.0
        pg_loss_reconst = 0.0
        pg_loss_d = 0.0
        J = len(c_trg_list)
        for j in range(J):
            with torch.no_grad():
                output1 = self.G(x_real, c_trg_list[j])
                c_reverse = c_trg_list[j].clone()
                for k in range(self.c_dim):
                    c_reverse[:, k] = (c_reverse[:, k] == 0)
            output2 = self.G(x_adv, c_trg_list[j])
            output3 = self.G(output2, c_org)
            out_src, out_cls = self.D(output2)

            with torch.no_grad():
                eta = torch.mean(torch.abs(output1 - x_real))
                eta_sum = eta_sum + eta

            dist_compare = torch.mean(torch.abs(output2 - output1))
            dist_reconst = torch.mean(torch.abs(output3 - x_adv))
            dist_d = torch.mean(out_src) + self.classification_loss(out_cls, c_reverse, self.dataset)

            # loss_fn = torch.nn.MSELoss()
            # dist_compare = loss_fn(output2, output1)
            # dist_reconst = loss_fn(output3, x_real)
            # dist_d = torch.mean(out_src)

            pg_loss_compare = pg_loss_compare + eta * dist_compare
            pg_loss_reconst = pg_loss_reconst + eta * dist_reconst
            pg_loss_d = pg_loss_d + dist_d

        # print(torch.mean(out_src))
        # print(self.classification_loss(out_cls, c_reverse, self.dataset))

        pg_loss_compare = pg_loss_compare / eta_sum
        pg_loss_reconst = pg_loss_reconst / eta_sum
        pg_loss_d = pg_loss_d / J

        output = self.PD(x_adv)
        pg_loss_fake = - torch.mean(output)

        pg_loss = - 10.0 * pg_loss_compare - 2.5 * pg_loss_reconst + pg_loss_d + 0.001 * pg_loss_fake
        # print(pg_loss_compare)
        # print(pg_loss_reconst)
        # print(pg_loss_d)
        # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

        # Logging.
        loss['PG/loss_compare'] = pg_loss_compare.item()
        loss['PG/loss_reconst'] = pg_loss_reconst.item()
        loss['PG/loss_d'] = pg_loss_d.item()
        loss['PG/loss_fake'] = pg_loss_fake.item()

        # Maximum the transfer loss, the reconstruction loss and the Discriminator loss at last iteration.
        if i > 0:
            for j in range(J):
                with torch.no_grad():
                    output1 = self.lastiter_G(x_real, c_trg_list[j])
                    c_reverse = c_trg_list[j].clone()
                    for k in range(self.c_dim):
                        c_reverse[:, k] = (c_reverse[:, k] == 0)
                output2 = self.lastiter_G(x_adv, c_trg_list[j])
                output3 = self.lastiter_G(output2, c_org)
                out_src, out_cls = self.lastiter_D(output2)

                with torch.no_grad():
                    eta = torch.mean(torch.abs(output1 - x_real))
                    eta_sum = eta_sum + eta

                dist_compare = torch.mean(torch.abs(output2 - output1))
                dist_reconst = torch.mean(torch.abs(output3 - x_adv))
                dist_d = torch.mean(out_src) + self.classification_loss(out_cls, c_reverse, self.dataset)

                pg_loss_compare = pg_loss_compare + eta * dist_compare
                pg_loss_reconst = pg_loss_reconst + eta * dist_reconst
                pg_loss_d = pg_loss_d + dist_d

            # print(torch.mean(out_src))
            # print(self.classification_loss(out_cls, c_reverse, self.dataset))

            pg_loss_compare = pg_loss_compare / eta_sum
            pg_loss_reconst = pg_loss_reconst / eta_sum
            pg_loss_d = pg_loss_d / J

            pg_loss_last_iter = - 10.0 * pg_loss_compare - 2.5 * pg_loss_reconst + pg_loss_d

            pg_loss = pg_loss + pg_loss_last_iter

        # Backward and optimize.
        self.pg_optimizer.zero_grad()
        pg_loss.backward()
        self.pg_optimizer.step()


        # =================================================================================== #
        #                                 4. Miscellaneous                                    #
        # =================================================================================== #

        self.PG.eval()
        dist_fn = torch.nn.MSELoss()

        # Print out training information.
        if (i + 1) % self.log_step == 0:
            et = time.time() - start_time
            et = str(datetime.timedelta(seconds=et))[:-7]
            log = "Elapsed [{}], Iteration [{}/{}]".format(et, i + 1, self.num_iters)
            for tag, value in loss.items():
                log += ", {}: {:.4f}".format(tag, value)
            print(log)

            if self.use_tensorboard:
                for tag, value in loss.items():
                    self.logger.scalar_summary(tag, value, i + 1)

        # Translate fixed images for debugging.
        if (i + 1) % self.sample_step == 0:
            with torch.no_grad():
                if (i + 1) == self.sample_step:
                    x_fake_list = [x_fixed]
                    for c_fixed in c_fixed_list:
                        x_fake_list.append(self.clean_G(x_fixed, c_fixed))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, 'clean_G/{}-clean-images.png'.format(i + 1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)

                x_fake_list = [x_fixed]
                for c_fixed in c_fixed_list:
                    x_fake_list.append(self.G(x_fixed, c_fixed))
                x_concat = torch.cat(x_fake_list, dim=3)
                sample_path = os.path.join(self.sample_dir, 'G/{}-clean-images.png'.format(i + 1))
                save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)

                x_fixed_ori = x_fixed
                pert = self.PG(x_fixed) * self.eps
                x_fixed = torch.clamp(x_fixed + pert, -1.0, 1.0)

                x_fake_list = [x_fixed]
                for c_fixed in c_fixed_list:
                    x_fake_list.append(self.clean_G(x_fixed, c_fixed))
                x_concat = torch.cat(x_fake_list, dim=3)
                sample_path = os.path.join(self.sample_dir, 'clean_G/{}-adv-images.png'.format(i + 1))
                save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)

                x_fake_list = [x_fixed]
                for c_fixed in c_fixed_list:
                    x_fake_list.append(self.G(x_fixed, c_fixed))
                x_concat = torch.cat(x_fake_list, dim=3)
                sample_path = os.path.join(self.sample_dir, 'G/{}-adv-images.png'.format(i + 1))
                save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)

                for j in range(len(x_fixed)):
                    print(dist_fn(x_fixed[j], x_fixed_ori[j]))

                print('Saved real and fake images into {}...'.format(sample_path))

        # Save model checkpoints.
        if (i + 1) % self.model_save_step == 0:
            G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i + 1))
            D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i + 1))
            PG_path = os.path.join(self.model_save_dir, '{}-PG.ckpt'.format(i + 1))
            PD_path = os.path.join(self.model_save_dir, '{}-PD.ckpt'.format(i + 1))
            torch.save(self.G.module.state_dict(), G_path)
            torch.save(self.D.module.state_dict(), D_path)
            torch.save(self.PG.module.state_dict(), PG_path)
            torch.save(self.PD.module.state_dict(), PD_path)
            print('Saved model checkpoints into {}...'.format(self.model_save_dir))

        # Decay learning rates.
        if (i + 1) % self.lr_update_step == 0 and (i + 1) > (self.num_iters - self.num_iters_decay):
            g_lr -= (self.g_lr / float(self.num_iters_decay))
            d_lr -= (self.d_lr / float(self.num_iters_decay))
            atk_lr -= (self.atk_lr / float(self.num_iters_decay))
            self.update_lr(g_lr, d_lr, atk_lr)
            print ('Decayed learning rates, g_lr: {}, d_lr: {}, atk_lr: {}.'.format(g_lr, d_lr, atk_lr))



        
        
    def attack(self):
        """Train PG against StarGAN within a single dataset."""
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_fixed, c_org = next(data_iter)
        x_fixed = x_fixed.to(self.device)
        c_fixed_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)
        
        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr
        atk_lr = self.atk_lr

        # Load the trained clean StarGAN.
        self.restore_clean_model(self.model_iters)
        
        # Start training from scratch.
        start_iters = 0
        
        # Build attack model and tgtmodel.
        self.PG = Attackmodel()
        self.PG = torch.nn.DataParallel(self.PG)
        self.print_network(self.PG, 'PG')
        self.PG.to(self.device)
        self.PD = PerturbationDiscriminator()
        self.PD = torch.nn.DataParallel(self.PD)
        self.PD.to(self.device)
        # self.tgtmodel = Attackmodel()
        # self.tgtmodel.to(self.device)
        # self.tgtmodel.load_state_dict(self.PG.state_dict())
        # self.tgt_optimizer = torch.optim.Adam(self.tgtmodel.parameters(), self.atk_lr, [self.beta1, self.beta2])
        self.pg_optimizer = torch.optim.Adam(self.PG.parameters(), self.atk_lr, [self.beta1, self.beta2])
        self.pd_optimizer = torch.optim.Adam(self.PD.parameters(), 0.1 * self.atk_lr, [self.beta1, self.beta2])
        
        # Start training.
        print('Start training...')
        start_time = time.time()
        for ii in range(start_iters, self.attack_iters):
            for i in range(start_iters, self.num_iters):
                    self.train_PG(d_lr, g_lr, atk_lr, data_loader, data_iter, x_fixed, c_fixed_list, start_time, i)
                    # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')



    # def blur_tensor(self, tensor):
    #     # preproc = smoothing.AverageSmoothing2D(channels=3, kernel_size=9).to(self.device)
    #     preproc = smoothing.GaussianSmoothing2D(sigma=0.75, channels=3, kernel_size=11).to(self.device)
    #     return preproc(tensor)



    def test(self):
        """Test batch images from the data loader under initiative defense or not."""
        # Load the trained generator.
        self.restore_clean_model(self.test_iters)
        if not os.path.exists('./defended_faces/'):
            os.makedirs('./defended_faces/')

        if self.use_PG:
            self.PG = Attackmodel()
            # self.path = 'stargan_celeba/models_'+str(self.eps)+'/30000-PG.ckpt'
            self.path = 'checkpoints/30000-PG-005.ckpt'
            self.PG.load_state_dict(torch.load(self.path))
            self.PG.to(self.device)
            self.PG.eval()
        
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader

        # defense loop.
        score = 0
        count = 0
        L1_sum = 0.0
        L2_sum = 0.0
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

                if self.use_PG:
                    pert = self.PG(x_real) * self.eps
                    x_adv = torch.clamp(x_real + pert, -1.0, 1.0)
                else:
                    x_adv = x_real

                # x_real = self.blur_tensor(x_real)
                # x_adv = self.blur_tensor(x_adv)

                # Translate images.
                x_fake_list = [x_adv]
                for c_trg in c_trg_list:
                    gen_clean = self.clean_G(x_real, c_trg)
                    gen_defended = self.clean_G(x_adv, c_trg)
                    x_fake_list.append(gen_defended)

                    L1_sum += F.l1_loss(gen_defended, gen_clean)
                    l2 = F.mse_loss(gen_defended, gen_clean)
                    L2_sum += l2
                    if l2 > 0.05:
                        score += 1
                    count += 1

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join('defended_faces/{}.png'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved defended and fake images into {}...'.format(result_path))

        print('L1 dist: %.6f'%(L1_sum/count))
        print('L2 dist: %.6f' % (L2_sum / count))
        print('DSR: %.6f'%(score*1.0/count))




    # def test_single_image(self):
    #     """Test single images from the data loader under initiative defense or not."""
    #     self.restore_clean_model(self.test_iters)

    #     if self.dataset == 'CelebA':
    #         data_loader = self.celeba_loader
    #     elif self.dataset == 'RaFD':
    #         data_loader = self.rafd_loader

    #     score = 0
    #     count = 0
    #     L1_sum = 0.0
    #     L2_sum = 0.0
    #     for i, (x_real, c_org) in enumerate(data_loader):

    #         x_real = x_real.to(self.device)
    #         c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

    #         self.adv_path = '/public/huangqidong/Na/stargan_celeba_256/results_test/adv/'+str(i+1)+'-images.png'
    #         x_adv = Image.open(self.adv_path)
    #         transform = []
    #         # transform.append(T.CenterCrop(crop_size))
    #         # transform.append(T.Resize(128))
    #         transform.append(T.ToTensor())
    #         transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    #         transform = T.Compose(transform)

    #         x_adv = transform(x_adv)
    #         x_adv = x_adv.unsqueeze(0)
    #         x_adv = x_adv.to(self.device)

    #         with torch.no_grad():
    #             x_fake_list = [x_adv]
    #             for c_fixed in c_trg_list:
    #                 genno = self.clean_G(x_real, c_fixed)
    #                 gen = self.clean_G(x_adv, c_fixed)
    #                 x_fake_list.append(gen)

    #                 L1_sum += F.l1_loss(gen, genno)
    #                 l2 = F.mse_loss(gen, genno)
    #                 L2_sum += l2
    #                 if l2 > 0.05:
    #                     score += 1
    #                 count += 1

    #         # Save the translated images.
    #         x_concat = torch.cat(x_fake_list, dim=3)
    #         result_path = '/public/huangqidong/Na/stargan_celeba_256/results_test/black-box/cnet+DD/'+str(i+1)+'-images.png'
    #         save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
    #         print('Saved real and fake images into {}...'.format(result_path))

    #     print('L1 dist: %.6f'%(L1_sum/count))
    #     print('L2 dist: %.6f' % (L2_sum / count))
    #     print('DSR: %.6f'%(score*1.0/count))



    # def test_multi(self):
    #     """Translate images using StarGAN trained on multiple datasets."""
    #     # Load the trained generator.
    #     self.restore_model(self.test_iters)
        
    #     with torch.no_grad():
    #         for i, (x_real, c_org) in enumerate(self.celeba_loader):

    #             # Prepare input images and target domain labels.
    #             x_real = x_real.to(self.device)
    #             c_celeba_list = self.create_labels(c_org, self.c_dim, 'CelebA', self.selected_attrs)
    #             c_rafd_list = self.create_labels(c_org, self.c2_dim, 'RaFD')
    #             zero_celeba = torch.zeros(x_real.size(0), self.c_dim).to(self.device)            # Zero vector for CelebA.
    #             zero_rafd = torch.zeros(x_real.size(0), self.c2_dim).to(self.device)             # Zero vector for RaFD.
    #             mask_celeba = self.label2onehot(torch.zeros(x_real.size(0)), 2).to(self.device)  # Mask vector: [1, 0].
    #             mask_rafd = self.label2onehot(torch.ones(x_real.size(0)), 2).to(self.device)     # Mask vector: [0, 1].

    #             # Translate images.
    #             x_fake_list = [x_real]
    #             for c_celeba in c_celeba_list:
    #                 c_trg = torch.cat([c_celeba, zero_rafd, mask_celeba], dim=1)
    #                 x_fake_list.append(self.G(x_real, c_trg))
    #             for c_rafd in c_rafd_list:
    #                 c_trg = torch.cat([zero_celeba, c_rafd, mask_rafd], dim=1)
    #                 x_fake_list.append(self.G(x_real, c_trg))

    #             # Save the translated images.
    #             x_concat = torch.cat(x_fake_list, dim=3)
    #             result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
    #             save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
    #             print('Saved real and fake images into {}...'.format(result_path))
