# This code is heavily based on Jan Mentzen's implementation of a VAE (https://jmetzen.github.io/2015-11-27/vae.html)

import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pickle as pkl

from torch.utils.data import Dataset, DataLoader
from torch.nn import Linear, ReLU, Module
from torch.nn.init import kaiming_uniform_, xavier_uniform_
import torch.nn as nn

np.random.seed(0)

import torch
import geomloss

from geomloss.utils import squared_distances

import argparse

parser = argparse.ArgumentParser(description='Process args for GAN.')
parser.add_argument('--weight_init', default = 'he')
parser.add_argument('--latent_dim', type = int, default = 32)
parser.add_argument('--lamb', type = float, default = .5)
args = parser.parse_args()


DIM = 64 # Model dimensionality (number of neurons in the hidden layer(s))
CRITIC_ITERS = 5 # How many critic iterations (Sinkhorn iterations) per generator iteration#was 50
BATCH_SIZE = 256 # Batch size
EPOCHS = 55#100000 # how many generator iterations to train for
DATA_DIM = 32
LATENT_DIM = args.latent_dim
INITIALIZATION = args.weight_init#'glorot'
COVARIANCE_SCALE = np.sqrt(DATA_DIM)
INITIALIZE_LAST = True
SAMPLE_SIZE = 100000
LAMBDA = args.lamb
MODE = "divergence"

def initializer(weight):
    if INITIALIZATION == 'he':
        return kaiming_uniform_(weight, nonlinearity='relu')
    if INITIALIZATION == 'glorot':
        return xavier_uniform_(weight)
    print("UNKNOWN INITIALIZATION")
    return None

def init_weights(m):
    if type(m) == nn.Linear:
        initializer(m.weight)
        
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
        return [np.random.randn(self.latent_dim).astype('float32'), self.X[index], 
                np.random.randn(self.latent_dim).astype('float32'), 
                np.random.randn(self.latent_dim).astype('float32')]
dataset = GaussianDataset()
data_dl = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)

class Generator(Module):
    def __init__(self, n_inputs = LATENT_DIM):
        super(Generator, self).__init__()

        self.model = nn.Sequential(        
        # defining the inputs to the first hidden layer
            Linear(n_inputs, DIM) ,
            ReLU(),
            Linear(DIM, DIM),
            ReLU(),
            Linear(DIM, DIM),
            ReLU(),
            Linear(DIM, DATA_DIM),
        )
        self.model.apply(init_weights)

    def forward(self, X):
        X = self.model(X)
        return X

class Discriminator(Module):
    def __init__(self, n_inputs = DATA_DIM):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(        
        # defining the inputs to the first hidden layer
            Linear(n_inputs, DIM) ,
            ReLU(),
            Linear(DIM, DIM),
            ReLU(),
            Linear(DIM, DIM),
            ReLU(),
            Linear(DIM, 1),
        )
        self.model.apply(init_weights)

    def forward(self, X):
        X = self.model(X)
        return X
    
# Initialize generator and discriminator
generator = Generator()
discriminator_fake = Discriminator() #for fake data (hat_Y) in W(Y, hat_Y)
discriminator_real = Discriminator() #for real data (Y) in W(Y, hat_Y)
discriminator_ff = Discriminator() #for fake data (hat_Y) in W(hat_Y, hat_Y)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr = 0.003)
optimizer_Df = torch.optim.RMSprop(discriminator_fake.parameters(),lr=0.005)
optimizer_Dr = torch.optim.RMSprop(discriminator_real.parameters(),lr=0.005)
optimizer_Dff = torch.optim.RMSprop(discriminator_ff.parameters(), lr = 0.01)
#optimizer_Dr = torch.optim.RMSprop()
#optimizer_Dff = torch.optim.RMSprop(discriminator_ff.parameters())

eps = LAMBDA
def sinkhorn_loss_dual(dual_fake, dual_real, cost_mat):#this is a negative loss
    log_pi = (dual_real.T+dual_fake - cost_mat)/eps
    reg_term = (log_pi).logsumexp((0,1))
    W_neg = - dual_fake.mean() - dual_real.mean() + eps*reg_term 
    return W_neg

res = {'loss':{}, 'sample': {}, 'cov_diff' : {}}
res['lambda'] = LAMBDA
cov_diff_all = []
epochs = EPOCHS


acc_cov_diff = []

cov_diff_inp = torch.tensor(np.random.randn(10*BATCH_SIZE, LATENT_DIM).astype('float32'))


epochs_passed = len(list(res['loss'].keys()))#to continue training 
loss_G = 0
for epoch in range(epochs_passed, epochs_passed+epochs):
    if epoch == 15:
            optimizer_G.param_groups[0]['lr'] = 0.001
    if epoch == 25:
            optimizer_G.param_groups[0]['lr'] = 0.0007
    if epoch == 35:
            optimizer_G.param_groups[0]['lr'] = 0.0005
            optimizer_Df.param_groups[0]['lr'] = 0.0025
            optimizer_Dr.param_groups[0]['lr'] = 0.0025
            optimizer_Dff.param_groups[0]['lr'] = 0.005
    for key in res:
        if key != 'lambda':
            res[key][epoch] = []
    # go through all the batches generated by dataloader
    for i, (inputs, real_sample, inputs0, inputs1) in enumerate(data_dl):
        # clear the gradients
        optimizer_Dr.zero_grad()
        optimizer_Df.zero_grad()
        optimizer_Dff.zero_grad()
        
        # compute the discriminator model outputs & update
        fake_sample = generator(inputs)
        
        cov_diff = np.linalg.norm(np.cov(np.array(fake_sample.data).T) - np.eye(DATA_DIM)/COVARIANCE_SCALE)
        cost_mat = squared_distances(fake_sample, real_sample)
        dual_fake = discriminator_fake(fake_sample)
        dual_real = discriminator_real(real_sample)
        loss_dual = sinkhorn_loss_dual(dual_fake, dual_real, cost_mat)
        loss_dual.backward()
        optimizer_Dr.step()
        optimizer_Df.step()
        
        #Sinkhorn negentropy update
        fake_sample0 = generator(inputs0)
        fake_sample1 = generator(inputs1)
        cost_mat_ff = squared_distances(fake_sample0, fake_sample1)
        dual_ff0 = discriminator_ff(fake_sample0)
        dual_ff1 = discriminator_ff(fake_sample1)
        loss_dual_f = sinkhorn_loss_dual(dual_ff0, dual_ff1, cost_mat_ff)
        loss_dual_f.backward()
        optimizer_Dff.step()
        
        if i % CRITIC_ITERS == 0 and i>0:
            # -----------------
            #  Train Generator
            # -----------------
            print("Generator step")
            optimizer_G.zero_grad()
            
            fake_sample = generator(inputs)
            cost_mat = squared_distances(fake_sample, real_sample)
            dual_fake = discriminator_fake(fake_sample)
            dual_real = discriminator_real(real_sample)
            loss_dual = sinkhorn_loss_dual(dual_fake, dual_real, cost_mat)
            fake_sample0 = generator(inputs0)
            fake_sample1 = generator(inputs1)
            cost_mat_ff = squared_distances(fake_sample0, fake_sample1)
            dual_ff0 = discriminator_ff(fake_sample0)
            dual_ff1 = discriminator_ff(fake_sample1)
            loss_dual_f = sinkhorn_loss_dual(dual_ff0, dual_ff1, cost_mat_ff)
            
            loss_G = - loss_dual + 0.5*loss_dual_f
            loss_G.backward()
            optimizer_G.step()
        if i % 50 == 0:
            fake_sample = generator(cov_diff_inp)
            acc_cov_diff += [np.linalg.norm(np.cov(np.array(fake_sample.data).T) - 
                                      np.eye(DATA_DIM)/COVARIANCE_SCALE)]
            
        print("Epoch: {}, iteration {}, loss {:.4f}, loss_f {:.4f}, , loss_G {:.4f}, cov_diff {}".format(epoch,
                                                                                        i, loss_dual, 
                                                                                       loss_dual_f, loss_G, 
                                                                                       cov_diff))
        res['loss'][epoch].append([loss_dual, loss_dual_f, loss_G])
        res['sample'][epoch].append(np.array(fake_sample.data))
        res['cov_diff'][epoch].append(cov_diff)
    cov_diff_all += res['cov_diff'][epoch]
    
if MODE == 'divergence':
    model_name = "P_SINKHORN_LAMBDA_"#P = parametric
else:
    model_name = "P_ENTROPIC_LAMBDA_"
model_name = model_name +str(LAMBDA) + "_LATENT_DIM_" + str(LATENT_DIM) + "_initialization_" + INITIALIZATION


torch.save(model.state_dict(), './models/'+model_name+'.pt')

plt.figure()
accuracy_history = np.array(acc_cov_diff[:400])
iteration = min(20000, len(accuracy_history)*50)
step = 50
plt.clf()
plt.grid("on", "both")
idx = np.max(np.where(accuracy_history > 3.5))
plt.plot(np.arange(idx*50,iteration,step), accuracy_history[idx:], linewidth=2, color = 'red')
plt.plot(np.arange(iteration+1), np.zeros(iteration+1), linewidth=2, color = 'green')
plt.savefig("./plots/accuracy_history_"+model_name+".png")

with open('./training_process_dump/' + model_name + '.pkl', 'wb') as f:
    pkl.dump([res,acc_cov_diff], f)
