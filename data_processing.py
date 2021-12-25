import torch
import numpy as np
import scipy.linalg


def generate_random_covariance(dim):
    '''
    Function that generates a random psd matrix of size dim x dim and Frobenius norm 1, the matrix is a product of a
    Gaussian matrix (W) and a uniform (0, 10) diagonal matrix
    :param dim: dimension of the output matrix
    :return:a random psd matrix normalized to have Frobenius norm 1
    '''
    W = np.random.randn(dim, dim)
    Sigma = np.diag(np.random.uniform(0, 10, dim))
    cov = W.dot(Sigma).dot(W.T)
    return cov / np.sqrt(np.sum(np.square(cov)))

class GaussianDataset(torch.utils.data.Dataset):
    def __init__(self, sample_size, data_dim, latent_dim, use_random_covariance=False, covariance_to_use=None):
        if covariance_to_use is not None and use_random_covariance:
            raise ValueError("Cannot use both random and fixed covariance matrix")
        # store the input and output features
        self.sample_size = sample_size
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.use_random_covariance = use_random_covariance

        np.random.seed(1)
        self.X = np.random.randn(sample_size, data_dim)
        if self.use_random_covariance:
            self.cov = generate_random_covariance(data_dim)
            self.sqrt_cov = scipy.linalg.sqrtm(self.cov)
        elif covariance_to_use is not None:
            self.cov = covariance_to_use
            self.sqrt_cov = scipy.linalg.sqrtm(self.cov)
        else:
            self.cov = np.eye(data_dim) / np.sqrt(data_dim)
            self.sqrt_cov = np.eye(data_dim) / np.sqrt(np.sqrt(data_dim))
        self.X = self.X.dot(self.sqrt_cov.T)
        self.latent_dim = latent_dim
        # ensure all data is numerical - type(float)
        self.X = self.X.astype('float32')

    # number of rows in dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, index):
        return self.X[index]