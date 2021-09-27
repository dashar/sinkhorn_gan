import os, sys

sys.path.append(os.getcwd())

import random
import matplotlib

#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
from torchcontrib.optim import SWA

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torch.nn.init import kaiming_uniform_, xavier_uniform_
import pickle as pkl
import geomloss #I changed the geomloss library!!!! (samples_loss.py and sinkhorn_divergence.py) to return 
                                                    #both dual potentials
torch.manual_seed(1)

import argparse

parser = argparse.ArgumentParser(description='Process args for GAN.')
parser.add_argument('--weight_init', default = 'he')
parser.add_argument('--latent_dim', type = int, default = 32)
parser.add_argument('--mode', default = 'sink-np')
parser.add_argument('--lamb', type = float, default = 1.8)#not using for now
args = parser.parse_args()


#TRAINING AND MODEL PARAMS
MODE = args.mode  # wgan-wc or wgan-gp or sink-p or sink-np
DIM = 64  # Model dimensionality
LAMBDA = args.lamb  # Smaller lambda seems to help for toy tasks specifically
CRITIC_ITERS = 5  # How many critic iterations per generator iteration
BATCH_SIZE = 256  # Batch size
EPOCHS = 150  # how many epochs to train for
LATENT_DIM = args.latent_dim
DATA_DIM = 32
SAMPLE_SIZE = 100000
COVARIANCE_SCALE = np.sqrt(DATA_DIM)
INITIALIZATION = args.weight_init
if MODE == 'sink-np':
    CRITIC_ITERS = 1
    BATCH_SIZE = 2560
    
    
class Generator(nn.Module):

    def __init__(self, n_inputs = LATENT_DIM, n_neurons = DIM, data_dim = DATA_DIM):
        super(Generator, self).__init__()

        main = nn.Sequential(
            nn.Linear(n_inputs, data_dim),
            #nn.ReLU(True),
            #nn.Linear(n_neurons, n_neurons),
            #nn.ReLU(True),
            #nn.Linear(n_neurons, n_neurons),
            #nn.ReLU(True),
            #nn.Linear(n_neurons, data_dim),
        )
        self.main = main

    def forward(self, noise):
        output = self.main(noise)
        return output


class Discriminator(nn.Module):

    def __init__(self, n_neurons = DIM, data_dim = DATA_DIM):
        super(Discriminator, self).__init__()

        main = nn.Sequential(
            nn.Linear(data_dim, n_neurons),
            nn.ReLU(True),
            nn.Linear(n_neurons, n_neurons),
            nn.ReLU(True),
            nn.Linear(n_neurons, n_neurons),
            nn.ReLU(True),
            nn.Linear(n_neurons, 1),
        )
        self.main = main

    def forward(self, inputs):
        output = self.main(inputs)
        return output.view(-1)
    
class WGAN():
    def __init__(self, n_neurons = DIM, data_dim = DATA_DIM, 
                 latent_dim = LATENT_DIM, gp = True, clip_value = (-.01,.01)):#gp=gradient penalty
        self.Gen = Generator(latent_dim, n_neurons, data_dim)
        self.Disc = Discriminator(n_neurons, data_dim)
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.n_neurons = n_neurons
        self.optimizerD = None
        self.optimizerG = None
        self.gp = gp
        self.clip_value = clip_value
        self.Discs = [self.Disc]
        
    def calc_gradient_penalty(self, real_data, fake_data):
        alpha = torch.rand(fake_data.shape[0], 1)
        alpha = alpha.expand(real_data.size())

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = self.Disc(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
        return gradient_penalty
    
    def disc_train(self, real_data):
        real_data_v = autograd.Variable(real_data)

        self.Disc.zero_grad()

        # train with real
        D_real = self.Disc(real_data_v)
        D_real = D_real.mean()

        # train with fake
        noise = torch.randn(real_data.shape[0], self.latent_dim)
        fake = self.Gen(noise).detach()
        inputv = fake
        D_fake = self.Disc(inputv)
        D_fake = D_fake.mean()
        if self.gp:
            # train with gradient penalty
            gradient_penalty = self.calc_gradient_penalty(real_data_v.data, fake.data)
        else:
            gradient_penalty = 0
            
        D_cost = D_fake - D_real + gradient_penalty
        D_cost.backward()#IF THIS DOESN'T work, change back to separate gradients
        Wasserstein_D = D_real - D_fake
        self.optimizerD.step()
        if not self.gp:
            # Clip weights of discriminator
            for p in self.Disc.parameters():
                p.data.clamp_(self.clip_value[0], self.clip_value[1])
                
        return D_cost.cpu().data.numpy()#, Wasserstein_D.cpu().data.numpy()
    
    def gen_train(self, real_data=None):#real data not actually used
        
        self.Gen.zero_grad()

        noise = torch.randn(BATCH_SIZE, self.latent_dim)
        noisev = autograd.Variable(noise)
        
        fake = self.Gen(noisev)
        G = self.Disc(fake)
        G = G.mean()
        G_cost = -G
        G_cost.backward()
        self.optimizerG.step()
        return G_cost.cpu().data.numpy(), fake.cpu().data.numpy()

def initializer(weight):
    if INITIALIZATION == 'he':
        return kaiming_uniform_(weight, nonlinearity='relu')
    if INITIALIZATION == 'glorot':
        return xavier_uniform_(weight)
    print("UNKNOWN INITIALIZATION")
    return None

# custom weights initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        initializer(m.weight)
        #m.bias.data.fill_(0)
        
def weights_init(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        m.bias.data.fill_(0.0)

# Dataset iterator
class GaussianDataset(Dataset):
    # reading the csv and defining predictor and output columns
    def __init__(self, sample_size=SAMPLE_SIZE, data_dim=DATA_DIM, latent_dim = LATENT_DIM,
                 variance = 1/COVARIANCE_SCALE):
    
        # store the input and output features
        
        np.random.seed(1)
        self.X = np.random.randn(sample_size,data_dim) * np.sqrt(variance)
        self.latent_dim = latent_dim
    
        # ensure all data is numerical - type(float)
        self.X = self.X.astype('float32')
    
    # number of rows in dataset
    def __len__(self):
        return len(self.X)
    
    # get a row at an index
    def __getitem__(self, index):
        return self.X[index]
    
def Sinkhorn_loss(x, y, epsilon, potentials = False):
    if not potentials:
        return 2 * geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=np.sqrt(epsilon / 2),
                                            debias=True)(x,y)
    f_x, f_y =  geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=np.sqrt(epsilon / 2),
                                            debias=True, potentials = True)(x,y)
    halved_potentials = list(f_x + f_y)
    potentials = [2 * f for f in halved_potentials]
    return potentials

class SGAN():#idea: learn the parametric discriminators using non-parametric dual potentials
    def __init__(self, n_neurons = DIM, data_dim = DATA_DIM, 
                 latent_dim = LATENT_DIM, epsilon = LAMBDA, param = True):#gp=gradient penalty
        self.Gen = Generator(latent_dim, n_neurons, data_dim)
        if param:
            self.Disc_real = Discriminator(n_neurons, data_dim)#real sample dual
            self.Disc_fake = Discriminator(n_neurons, data_dim)#fake sample dual
            self.Disc_negent = Discriminator(n_neurons, data_dim)#for negentropy dual
            self.optimizerD = None
            self.Discs = [self.Disc_real, self.Disc_fake, self.Disc_negent]
        else:
            self.Discs = []
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.n_neurons = n_neurons
        self.optimizerG = None
        self.param = param
        self.epsilon = epsilon
        #for the logs
        if param:
            self.metrics_names = ['ent W2 cost', 'negent cost', 'gen cost', 'cov diff', 'ent W2 cost test',
                             'negent cost test', 'gen cost test', 'cov diff test']
        else:
            self.metrics_names = ['Sink cost', 'cov diff', 'Sink cost test', 'cov diff test']
        self.logs = {key:{} for key in self.metrics_names}
        self.logs['lambda'] = epsilon
        
    
    def disc_train(self, real_data, cov_diff_fn, epoch=0, iteration=0):
        if not self.param:
            return None
        
        for disc in self.Discs:
            disc.zero_grad()

        # real sample dual potential
        D_real = self.Disc_real(real_data)

        # fake sample dual potential
        noise = torch.randn(real_data.shape[0], self.latent_dim)
        fake_data = self.Gen(noise).detach()
        D_fake = self.Disc_fake(fake_data)
        
        # negentropy dual potential
        D_negent = self.Disc_negent(fake_data)
        
        #negative entropic W2
        cost_mat = (real_data.view(-1,1,DATA_DIM) - fake_data.view(1,-1,DATA_DIM)).pow(2).sum(2)
        neg_loss_entropic = -D_real.mean()-D_fake.mean() \
                + self.epsilon * ((D_real.view(-1,1)+D_fake.view(1,-1) - cost_mat)/self.epsilon).logsumexp((0,1))#exp().mean()
        
        #negentropy 
        cost_mat_fake = (fake_data.view(-1,1,DATA_DIM) - fake_data.view(1,-1,DATA_DIM)).pow(2).sum(2)
        loss_negent = -2. * D_negent.mean()\
          + self.epsilon * ((D_negent.view(-1,1)+D_negent.view(1,-1) - cost_mat_fake)/self.epsilon).logsumexp((0,1))#exp().mean()
        neg_loss_entropic.backward()
        loss_negent.backward()
        #(loss_negent + neg_loss_entropic).backward() #this loss function as the gradients are computed for non-intersecting parameters
        self.optimizerD.step()
        #for i in range(len(self.optimizersD)):
        #    self.optimizersD[i].step()
        
        #Sinkhorn_D = -loss_entropic + 0.5 * loss_negent
        self.logs['negent cost'][epoch][iteration] = loss_negent.data.numpy()
        self.logs['ent W2 cost'][epoch][iteration] = neg_loss_entropic.data.numpy()
                
        return [neg_loss_entropic.cpu().data.numpy(), loss_negent.cpu().data.numpy()]
    
    def disc_loss(self, noise, real_data, epoch=0, iteration=0):
        if not self.param:
            return None
        
        # real sample dual potential
        D_real = self.Disc_real(real_data)
        D_real = D_real

        # fake sample dual potential
        fake_data = self.Gen(noise)
        D_fake = self.Disc_fake(fake_data)
        
        # negentropy dual potential
        D_negent = self.Disc_negent(fake_data)
        
        #negative entropic W2#if there are issues try the log (negative)
        cost_mat = (real_data.view(-1,1,DATA_DIM) - fake_data.view(1,-1,DATA_DIM)).pow(2).sum(2)
        neg_loss_entropic = -D_real.mean()-D_fake.mean() \
                + self.epsilon * ((D_real.view(-1,1)+D_fake.view(1,-1) - cost_mat)/self.epsilon).logsumexp((0,1))#exp().mean()
        #neg_loss_entropic.backward()
        #self.optimizersD[0].step()
        
        #negentropy 
        cost_mat_fake = (fake_data.view(-1,1,DATA_DIM) - fake_data.view(1,-1,DATA_DIM)).pow(2).sum(2)
        loss_negent = -2. * D_negent.mean()\
          + self.epsilon * ((D_negent.view(-1,1)+D_negent.view(1,-1) - cost_mat_fake)/self.epsilon).logsumexp((0,1))#exp().mean()
                
        self.logs['negent cost test'][epoch][iteration] = loss_negent.data.numpy()
        self.logs['ent W2 cost test'][epoch][iteration] = neg_loss_entropic.data.numpy()
        
    
    def gen_train(self, cov_diff_fn, real_data=None, epoch = 0, iteration = 0):
        
        self.Gen.zero_grad()
        if real_data is not None:
            noise = torch.randn(real_data.shape[0], self.latent_dim)
        else:
            noise = torch.randn(BATCH_SIZE, self.latent_dim)
        fake_data = self.Gen(noise)
        
        if not self.param:
            #f_x, f_y = geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=np.sqrt(self.epsilon),
            #                            debias=True, potentials=True)(fake_data,real_data)
            #dual_x, dual_xx = f_x
            #dual_y, dual_yy = f_y
            dual_x, dual_xx, dual_y, dual_yy = Sinkhorn_loss(fake_data, real_data, 
                                                             self.epsilon, potentials = True)
        else:
            dual_x = self.Disc_fake(fake_data)
            dual_xx = self.Disc_negent(fake_data)
            dual_y = self.Disc_real(real_data)
            
            
        #cost matrices (were previously /2)
        C_xx = (fake_data.view(-1, 1, self.data_dim)
                    - fake_data.view(1,-1,self.data_dim)).square().sum(2)

        C_xy = (fake_data.view(-1, 1, self.data_dim) 
                    - real_data.view(1,-1,self.data_dim)).square().sum(2)

        #probabilities
        N = real_data.shape[0]
        pi_xx_log = ((dual_xx.view(-1, 1) + dual_xx.view(1,-1)) - C_xx) / self.epsilon
        pi_xx = (pi_xx_log - pi_xx_log.max()).exp()
        pi_xx_log -= pi_xx.sum().log()
        pi_xx /= pi_xx.sum()
        
        pi_xy_log = ((dual_x.view(-1, 1) + dual_y.view(1,-1)) - C_xy) / self.epsilon
        pi_xy = (pi_xy_log - pi_xy_log.max()).exp()
        pi_xy_log -= pi_xy.sum().log()
        pi_xy /= pi_xy.sum()
        
        grad = (pi_xx.view(N, N, 1) * fake_data.view(1, N, self.data_dim) - 
                    pi_xy.view(N, N, 1) * real_data.view(1, N, self.data_dim)).sum(1).detach()
        loss = (grad * fake_data).sum()
        loss.backward()
        self.optimizerG.step()
        
        if not self.param:
            Sinkhorn_D = (dual_x - dual_xx).mean() + (dual_y - dual_yy).mean()
            self.logs['Sink cost'][epoch][iteration] = Sinkhorn_D.data.numpy()
            
        
        else:#MAYBE CHANGE LOSS CALCULATION HERE TO MATCH WHAT WAS IN "ON THE CONVERGENCE AND ROBUSTNESS..."
            
            #ASSUMING OPTIMAL DUAL POTENTIALS:
            #Sinkhorn_D = dual_x.mean() - dual_xx.mean()
        
            #Sinkhorn_D.backward()
            #self.optimizerG.step()
            
            Sinkhorn_D = (pi_xy * C_xy - self.epsilon * pi_xy * pi_xy_log).sum() - \
                0.5 * (pi_xx * C_xx - self.epsilon * pi_xx * pi_xx_log).sum()# - \
                #0.5 * Sinkhorn_loss(real_data, real_data, self.epsilon)#last term not necessary
            
            self.logs['gen cost'][epoch][iteration] = Sinkhorn_D.data.numpy()
        
        
        self.logs['cov diff'][epoch][iteration] = cov_diff_fn(fake_data.data.numpy())
        return Sinkhorn_D.cpu().data.numpy(), fake_data.cpu().data.numpy()
        
    
    def gen_loss(self, noise, cov_diff_fn, real_data=None, epoch = 0, iteration = 0):
        
        self.Gen.zero_grad()
        fake_data = self.Gen(noise)
        
        if not self.param:
            Sinkhorn_D = Sinkhorn_loss(fake_data, real_data, self.epsilon)
            self.logs['Sink cost test'][epoch][iteration] = Sinkhorn_D.data.numpy()
            
        
        else:#MAYBE CHANGE LOSS CALCULATION HERE TO MATCH WHAT WAS IN "ON THE CONVERGENCE AND ROBUSTNESS..."
            dual_x = self.Disc_fake(fake_data)
            dual_xx = self.Disc_negent(fake_data)
            dual_y = self.Disc_real(real_data)
            #cost matrices (were previously /2)
            C_xx = (fake_data.view(-1, 1, self.data_dim)
                        - fake_data.view(1,-1,self.data_dim)).square().sum(2)

            C_xy = (fake_data.view(-1, 1, self.data_dim) 
                        - real_data.view(1,-1,self.data_dim)).square().sum(2)

            #probabilities
            N = real_data.shape[0]
            pi_xx_log = ((dual_xx.view(-1, 1) + dual_xx.view(1,-1)) - C_xx) / self.epsilon
            pi_xx = (pi_xx_log - pi_xx_log.max()).exp()
            pi_xx_log -= pi_xx.sum().log()
            pi_xx /= pi_xx.sum()

            pi_xy_log = ((dual_x.view(-1, 1) + dual_y.view(1,-1)) - C_xy) / self.epsilon
            pi_xy = (pi_xy_log - pi_xy_log.max()).exp()
            pi_xy_log -= pi_xy.sum().log()
            pi_xy /= pi_xy.sum()

            Sinkhorn_D = (pi_xy * C_xy - self.epsilon * pi_xy * pi_xy_log).sum() - \
                0.5 * (pi_xx * C_xx - self.epsilon * pi_xx * pi_xx_log).sum() #- \
                #0.5 * Sinkhorn_loss(real_data, real_data, self.epsilon)#last term not necessary
            self.logs['gen cost test'][epoch][iteration] = Sinkhorn_D.data.numpy()
        
        
        self.logs['cov diff test'][epoch][iteration] = cov_diff_fn(fake_data.data.numpy())
        return Sinkhorn_D.cpu().data.numpy(), fake_data.cpu().data.numpy()
    
    
class cov_metric():
    def __init__(self, data_dim = DATA_DIM, average = False):
        if average:
            self.cov = np.zeros((data_dim,data_dim))
            self.n_seen = 0
            self.mean = np.zeros(data_dim)
        self.optimal_cov = np.eye(data_dim) / COVARIANCE_SCALE
        self.data_dim = data_dim
        self.average = average
    def update(self, fake_data):
        sample_cov = np.cov(fake_data.T)
        if self.average:
            data_frac = self.n_seen / (self.n_seen + fake_data.shape[0])
            sample_mean = np.mean(fake_data, axis = 0)
            self.cov = data_frac * (self.cov + np.outer(self.mean, self.mean)) \
                    + (1 - data_frac) * (sample_cov + np.outer(sample_mean, sample_mean))
            self.mean = data_frac * self.mean + (1 - data_frac) * sample_mean
        else:
            self.cov = sample_cov
        return np.linalg.norm(self.cov - self.optimal_cov)
    
def plot_logs(key, logs):
    y = []#key
    x = []#iterations
    iters_passed = 0
    for epoch in range(max(logs[key].keys())+1):
        for it in sorted(list(logs[key][epoch].keys())):
            y += [logs[key][epoch][it]]
            x += [it + iters_passed]
        iters_passed += max(logs[key][epoch].keys())+1
        
    return x, y
    
if MODE == 'wgan-gp':
    model_name = "WGAN_GP"
elif MODE == 'wgan-wc':
    model_name = "WGAN_WC"
elif MODE == 'sink-np':
    model_name = "SINKHORN_NP"
elif MODE == 'sink-p':
    model_name = "SINKHORN_P"
model_name = model_name + "_LATENT_DIM_" + str(LATENT_DIM) + "_initialization_" + INITIALIZATION + "_LAMBDA_" + str(LAMBDA)
model_name += '_v2_linear'

dataset = GaussianDataset()
data_dl = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)


if MODE == 'wgan-gp':
    model = WGAN()
elif MODE == 'wgan-wc':
    model = WGAN(gp = False)
elif MODE == 'sink-np':
    model = SGAN(param = False)
elif MODE == 'sink-p':
    model = SGAN()
    
model.Gen.apply(weights_init)
for disc in model.Discs:
    disc.apply(weights_init);
    
if MODE == 'wgan-gp':   
    model.optimizerD = optim.Adam(model.Disc.parameters(), lr=1e-4, betas=(0.5, 0.9))
    model.optimizerG = optim.Adam(model.Gen.parameters(), lr=1e-4, betas=(0.5, 0.9))
elif MODE == 'wgan-wc':
    model.optimizerD = optim.RMSprop(model.Disc.parameters(), lr=5e-5)
    model.optimizerG = optim.RMSprop(model.Gen.parameters(), lr=5e-5)
elif MODE == 'sink-p':
    model.optimizerD = optim.Adam(list(model.Discs[0].parameters()) 
                             + list(model.Discs[1].parameters()) 
                                    +list(model.Discs[2].parameters()), lr=1e-4, betas=(0.5, 0.9))
    model.optimizerG = optim.Adam(model.Gen.parameters(), lr=1e-4, betas=(0.5, 0.9))
elif MODE == 'sink-np':
    model.optimizerG = optim.SGD(model.Gen.parameters(), lr=1., momentum = 0.5)
    #optim.Adam(model.Gen.parameters(), lr=1e-4, betas=(0.5, 0.9))
    
average_cov = False
model.logs['batch size'] = BATCH_SIZE
model.logs['critic iters'] = CRITIC_ITERS
model.logs['mode'] = MODE
model.logs['lambda'] = LAMBDA

if 'weights' not in model.logs:
    model.logs['weights'] = {}

    
real_data_test = torch.FloatTensor(np.random.randn(2560, 32) / np.sqrt(COVARIANCE_SCALE))
noise_test = torch.FloatTensor(np.random.randn(2560, 32))
plt.figure()

def lr_scheduler(epoch, model):
    if epoch < 10:
        model.optimizerG.param_groups[0]['lr'] = 1. #0.27 test covariance diff 0.267 and 0.235 test Sink div
    elif epoch <20:
        model.optimizerG.param_groups[0]['lr'] = 0.5 #0.267 cov 0.235 Sink div
    elif epoch <30:
        model.optimizerG.param_groups[0]['lr'] = 0.25 #0.267 cov 0.235 Sink div
    elif epoch <40:
        model.optimizerG.param_groups[0]['lr'] = 0.175 #0.267 cov 0.235 Sink div
    elif epoch <50:
        model.optimizerG.param_groups[0]['lr'] = 0.09 #0.267 cov 0.235 Sink div
    elif epoch < 60:
        model.optimizerG.param_groups[0]['lr'] = 0.05 #0.267 cov 0.235 Sink div
    elif epoch < 80:
        model.optimizerG.param_groups[0]['lr'] = 0.025#0.267 cov 0.235 Sink div
    elif epoch < 100:
        model.optimizerG.param_groups[0]['lr'] = 0.01#0.267 cov 0.235 Sink div
    elif epoch < 120:
        model.optimizerG.param_groups[0]['lr'] = 0.005#0.267 cov 0.235 Sink div
    elif epoch == 120:
        base_opt = optim.SGD(model.Gen.parameters(), lr=5e-3)#, momentum = 0.9)
        model.optimizerG = SWA(base_opt, swa_start=0, swa_freq=1, swa_lr=0.005)
    #return 0.01

    
if 'weights' not in model.logs:
    model.logs['weights'] = {}
for epoch in range(EPOCHS):
    cov_diff_fn = cov_metric(average=average_cov)
    #if epoch > 0:
    for key in model.metrics_names:
        model.logs[key][epoch] = {}
    model.logs['weights'][epoch] = {}
    for iteration, _data in enumerate(data_dl):
        if _data.shape[0] !=BATCH_SIZE:
            continue
        lr_scheduler(epoch, model)
        
        ############################
        # (1) Update D network CRITIC_ITERS iterations
        ###########################
        model.disc_train(_data, cov_diff_fn.update, epoch=epoch, iteration=iteration)
        model.disc_loss(noise_test,real_data_test, epoch=epoch, iteration=iteration)
        if iteration % CRITIC_ITERS == CRITIC_ITERS - 1 :
            ############################
            # (2) Update G network
            ###########################
            #ms = model.optimizerG.state_dict()
            model.gen_train(real_data=_data, cov_diff_fn=cov_diff_fn.update, epoch=epoch, iteration=iteration)
            model.gen_loss(noise_test, cov_diff_fn.update, real_data_test, epoch, iteration)
            metrics = [model.logs[key][epoch][iteration] for key in model.metrics_names]
            print(('epoch: {}, iter: {}, '+': {:.5f}, '.join(model.metrics_names) + ': {:.5f},').format(epoch,
                                                                                iteration,*metrics))
            model.logs['weights'][epoch][iteration] = [model.Gen.state_dict()]+ \
                                        [disc.state_dict() for disc in model.Discs]
    if epoch >= 120:
        model.optimizerG.swap_swa_sgd()
        
    if (epoch+1) % 5 == 0:
        with open('./training_process_dump/' + model_name + '.pkl', 'wb') as f:
            pkl.dump(model.logs, f)
            
        plt.clf()
        plt.grid("on", "both")
        x, y = plot_logs('cov diff test', model.logs)
        idx = np.max(np.where(np.array(y) > 3.5))+1
        plt.plot(x[idx:], y[idx:])
        plt.plot(x, np.zeros(len(x)))
        plt.savefig("./plots/accuracy_history_"+model_name+".png")


