import numpy as np
import matplotlib
#matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pickle as pkl

from torch.utils.data import Dataset, DataLoader
from torch.nn import Linear, ReLU
from torch.nn.init import kaiming_uniform_, xavier_uniform_

np.random.seed(0)

import torch
import geomloss

import argparse

parser = argparse.ArgumentParser(description='Process args for GAN.')
parser.add_argument('--weight_init', default = 'he')
parser.add_argument('--latent_dim', type = int, default = 32)
parser.add_argument('--lamb', type = float, default = .5)
args = parser.parse_args()


DIM = 64 # Model dimensionality (number of neurons in the hidden layer(s))
CRITIC_ITERS = 5 # How many critic iterations (Sinkhorn iterations) per generator iteration#was 50
BATCH_SIZE = 256 # Batch size
EPOCHS = 550#100000 # how many generator iterations to train for
DATA_DIM = 32
LATENT_DIM = args.latent_dim
INITIALIZATION = args.weight_init#'glorot'
COVARIANCE_SCALE = np.sqrt(DATA_DIM)
INITIALIZE_LAST = True
SAMPLE_SIZE = 100000
LAMBDA = args.lamb
MODE = 'divergence'#args.loss #'loss'

if MODE == 'divergence':
    model_loss = geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=np.sqrt(LAMBDA), debias=True)
elif MODE == 'loss':
    model_loss = geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=np.sqrt(LAMBDA), debias=False)
else:
    print("MODE UNKNOWN")

def initializer(weight):
    if INITIALIZATION == 'he':
        return kaiming_uniform_(weight, nonlinearity='relu')
    if INITIALIZATION == 'glorot':
        return xavier_uniform_(weight)
    print("UNKNOWN INITIALIZATION")
    return None

class Generator(torch.nn.Module):
    def __init__(self, n_inputs = LATENT_DIM):
        # calling constructor of parent class
        super().__init__()

        # defining the inputs to the first hidden layer
        self.hid1 = Linear(n_inputs, DIM) 
        initializer(self.hid1.weight)
        self.act1 = ReLU()

        # defining the inputs to the second hidden layer
        self.hid2 = Linear(DIM, DIM)
        initializer(self.hid2.weight)
        self.act2 = ReLU()

        # defining the inputs to the third hidden layer
        self.hid3 = Linear(DIM, DIM)
        initializer(self.hid3.weight)
        self.act3 = ReLU()

        # defining the inputs to the output layer
        self.out = Linear(DIM, DATA_DIM)
        initializer(self.out.weight)

    def forward(self, X):
        #input and act for layer 1
        X = self.hid1(X)
        X = self.act1(X)
        
        #input and act for layer 2
        X = self.hid2(X)
        X = self.act2(X)
        
        #input and act for layer 3
        X = self.hid3(X)
        X = self.act3(X)
        
        #input for output layer
        X = self.out(X)
        
        return X

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
        return [np.random.randn(self.latent_dim).astype('float32'), self.X[index]]
    
dataset = GaussianDataset()
data_dl = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)

# define the network
model = Generator(LATENT_DIM)
# define the number of epochs
epochs = EPOCHS
# define the optimizer - Adam
optimizer = torch.optim.Adam(model.parameters(), lr = 0.005)
# define the loss function
criterion = model_loss

res = {'loss':{}, 'sample': {}, 'cov_diff' : {}}
res['lambda'] = LAMBDA
cov_diff_all = []

plt.figure()
if MODE == 'divergence':
    model_name = "NP_SINKHORN_LAMBDA_"#non parametric
else:
    model_name = "NP_ENTROPIC_LAMBDA_"
model_name = model_name +str(LAMBDA) + "_LATENT_DIM_" + str(LATENT_DIM) + "_initialization_" \
    + INITIALIZATION + "_BATCHSIZE" + str(BATCH_SIZE)


acc_cov_diff = []
cov_diff_inp = torch.tensor(np.random.randn(4*BATCH_SIZE, LATENT_DIM).astype('float32'))


# iterate through all the epoch
epochs_passed = len(list(res['loss'].keys()))#to continue training 
for epoch in range(epochs_passed, epochs_passed+epochs):
    for key in res:
        if key != 'lambda':
            res[key][epoch] = []
    #if epoch % 5 == 0 and epoch > 0 and LAMBDA == 0.5:
    #    optimizer.param_groups[0]['lr'] /= 1.2
    if epoch % 5 == 0 and epoch > 0 and LAMBDA == 2.0:
        optimizer.param_groups[0]['lr'] /= 1.03
    # go through all the batches generated by dataloader
    for i, (inputs, targets) in enumerate(data_dl):
        # clear the gradients
        optimizer.zero_grad()
        # compute the model output
        yhat = model(inputs)
        # calculate loss
        loss = criterion(yhat, targets)
        cov_diff = np.linalg.norm(np.cov(np.array(yhat.data).T) - np.eye(DATA_DIM)/COVARIANCE_SCALE)
        # credit assignment
        loss.backward()
        # update model weights
        optimizer.step()
        yhat = model(inputs)
        cov_diff2 = np.linalg.norm(np.cov(np.array(yhat.data).T) - np.eye(DATA_DIM)/COVARIANCE_SCALE)
        print("Epoch: {}, iteration {}, loss {}, cov_diff {} cov_diff train {}".format(epoch,i, loss, 
                                                                                       cov_diff, cov_diff2))
        res['loss'][epoch].append(loss.detach().cpu().numpy())
        res['sample'][epoch].append(yhat.detach().cpu().numpy())
        res['cov_diff'][epoch].append(cov_diff)
        
        if i % 50 == 0:
            fake_sample = model(cov_diff_inp)
            acc_cov_diff += [np.linalg.norm(np.cov((fake_sample.detach().cpu().numpy()).T) - 
                                      np.eye(DATA_DIM)/COVARIANCE_SCALE)]
            
    cov_diff_all += res['cov_diff'][epoch]
    with open('./training_process_dump/' + model_name + '.pkl', 'wb') as f:
        pkl.dump(res, f)
    
    plt.clf()
    plt.grid("on", "both")
    plt.plot(cov_diff_all)
    plt.plot(np.arange(len(cov_diff_all)), np.zeros(len(cov_diff_all)))
    plt.savefig("./plots/accuracy_history_"+model_name+".png")
    
#step = 10
#plt.clf()
#idx = np.max(np.where(np.array(cov_diff_all) > 3.5))
#plt.grid("on", "both")
#plt.plot(np.arange(idx, len(cov_diff_all), step), cov_diff_all[idx::step])
#plt.plot(np.arange(len(cov_diff_all)), np.zeros(len(cov_diff_all)))
#plt.savefig("./plots/accuracy_history_"+model_name+".png")

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