import numpy
import numpy as np
import torch


class CovarianceMatrixCalculator:
    """
    Class to calculate covariance matrix as a running estimate of the values passed to update function
    """
    def __init__(self, data_dim=None):
        """
        Initialize the covariance_matrix_calculator class member
        :param data_dim: the data dimension, if None, will be inferred from the first call to update
        """
        self.data_dim = data_dim
        # Initialize the moments to None before they are used
        self.second_moment = None
        self.mean = None
        self.covariance = None
        self.number_of_seen_samples = 0
        self.latest_batch_covariance = None
        self.updated = False

    def update(self, data_batch, return_batch_covariance=False):
        """
        Update and output the covariance matrix
        :param data_batch: a Nxd tensor containing N data points of dimension d
        :param return_batch_covariance: (boolean) if True, the covariance of data_batch will be returned (but the
                                        running covariance will still be updated if  self.updated is false, otherwise
                                        the running covariance is returned
        :return: sample covariance matrix
        """
        if self.updated:
            return self.latest_batch_covariance if return_batch_covariance else self.covariance
        self.updated = True
        data_batch = data_batch.cpu().detach().numpy()
        batch_size = data_batch.shape[0]
        seen_data_fraction = self.number_of_seen_samples / (self.number_of_seen_samples + batch_size)
        sample_mean = np.mean(data_batch, axis=0)
        sample_second_moment = data_batch.T.dot(data_batch / batch_size)
        self.latest_batch_covariance = (sample_second_moment - np.outer(sample_mean, sample_mean))/(1 - 1/batch_size)
        if self.data_dim is None or self.second_moment is None:
            self.data_dim = data_batch.shape[1]
            self.second_moment = sample_second_moment
            self.mean = sample_mean
            self.covariance = (sample_second_moment - np.outer(self.mean, self.mean))
            if batch_size > 1:
                self.covariance *= batch_size / (batch_size - 1)
            self.number_of_seen_samples = batch_size
            return self.latest_batch_covariance if return_batch_covariance else self.covariance
        self.second_moment *= seen_data_fraction
        self.second_moment += sample_second_moment * (1 - seen_data_fraction)
        self.mean = seen_data_fraction * self.mean + (1 - seen_data_fraction) * sample_mean
        self.number_of_seen_samples += batch_size
        self.covariance = (self.second_moment - np.outer(self.mean, self.mean)) * \
                          (self.number_of_seen_samples/(self.number_of_seen_samples - 1))
        return self.covariance

    def set_updated_to_false(self):
        """
        Set the 'updated' attribute to False (Needed to be updating the moments only once when update is run at
        calculations of different metrics)
        :return:
        """
        self.updated = False

    def reset(self):
        """
        Sets the calculated moments to None
        """
        self.second_moment = None
        self.mean = None
        self.covariance = None
        self.number_of_seen_samples = 0
        self.latest_batch_covariance = None
        self.updated = False


def get_optimal_output_covariance_gaussian_sgan(data_covariance, latent_dim):
    eigenvals, eigenvecs = np.linalg.eigh(data_covariance)
    eigenvals[:-latent_dim] = 0
    return eigenvecs.dot(eigenvals[:, np.newaxis] * eigenvecs.T)

def get_optimal_output_covariance_gaussian_ew2gan(data_covariance, latent_dim, epsilon):
    eigenvals, eigenvecs = np.linalg.eigh(data_covariance)
    eigenvals[:-latent_dim] = 0
    eigenvals = np.maximum(eigenvals - epsilon/2, 0)
    return eigenvecs.dot(eigenvals[:, np.newaxis] * eigenvecs.T)


def get_optimal_output_covariance_gaussian_entropic_wgan(data_covariance, latent_dim, regularization_strength):
    eigenvals, eigenvecs = np.linalg.eigh(data_covariance)
    eigenvals[:-latent_dim] = 0
    eigenvals = np.maximum(eigenvals - regularization_strength/2, 0)
    return eigenvecs.dot(eigenvals[:, np.newaxis] * eigenvecs.T)


def frobenius_distance_to_generator_output_covariance(target_matrix, generator_output, covariance_calculator, GAN,
                                                      permute_target_matrix_diagonal=False,
                                                      use_sample_covariance=False):
    """
    Calculates Frobenius distance between the target_matrix and the covariance matrix of the generator output. Average
    covariance matrix of the generator output is supported via the covariance_calculator, but for a linear generator
    the weight product is used instead of an averaged covariance matrix
    :param target_matrix: matrix to compute the distance to (can be optimal or true covariance matrix)
    :param generator_output: the output of the generator on a batch of inputs, a tensor of size batch_size x dim
    :param covariance_calculator:a CovarianceMatrixCalculator member that keeps track of an average, can be None for a
                                 linear generator, otherwise an error will be thrown
    :param GAN:a GAN model that will be associated with the metric
    :param permute_target_matrix_diagonal: whether to permute the target matrix diagonal to ge a smaller distance
                                            (for a W2GAN or an SGAN, if the target  distribution has covariance
                                            proportional to identity, we the optimal covariace's  diagonal can be
                                            permuted resulting in a new optimal covariance. In general, if the
                                            smallest non-zero eigenvalue of the optimal covariance has multiplicity
                                            > 1 and its multiplicity on the optimal covariance is smaller than in the
                                            target distribution covariance (both for SGAN, entropic W2GAN and W2GAN),
                                            meaning that one of the equal eigenvalues was made zero, then the subspace
                                            of the zero-ed eigenvalue can be chosen so that it gives the smallest
                                            frobenius norm. In case of a matrix proportional to identity this is
                                            eqivalent to diagonal permutation of the target_matrix
    :param use_sample_covariance: (boolean) whether to use sample covariance (if True) or use the average covariance
                                            calculated with covariance_calculator in case of a non-linear generator or
                                            the weights dot product in case of a linear generator
    :return:
    """
    if use_sample_covariance:
        if covariance_calculator is not None:
            covariance_calculator.update(generator_output)
            generator_output_covariance = covariance_calculator.latest_batch_covariance
        else:
            generator_output_covariance = np.cov(generator_output.cpu().detach().numpy().T)
    else:
        if GAN.generator.linear:
            weight = GAN.generator.get_parameter('main.dense_0.weight').cpu().detach().numpy()
            generator_output_covariance = weight.dot(weight.T)
            # Update the covariance calculator even  if it is not used here
            if covariance_calculator is not None:
                covariance_calculator.update(generator_output)
        elif covariance_calculator is None:
            raise ValueError("Covariance calculator is None and an average covariance is used for non-linear generator")
        else:
            generator_output_covariance = covariance_calculator.update(generator_output)
    if permute_target_matrix_diagonal:
        idx = np.arange(generator_output_covariance.shape[0])
        difference_without_diagonal = generator_output_covariance - target_matrix
        difference_without_diagonal[idx, idx] = 0
        diagonal_difference_with_permutation = np.sort(generator_output_covariance.diagonal()) - \
                                               np.sort(target_matrix.diagonal())
        squared_frobenius_distance = np.sum(np.square(difference_without_diagonal)) + \
                                     np.sum(np.square(diagonal_difference_with_permutation))
        return np.sqrt(squared_frobenius_distance)
    return np.linalg.norm(target_matrix - generator_output_covariance)


def create_all_covariance_metrics(dataset_covariance, covariance_calculator, gan, latent_dim,
                                  permute_target_matrix_diagonal=False, SGAN_model=True, epsilon=None):
    """
    A function that creates 4 covariance difference metrics: 2 for the running covariance and 2 for the batch covariance
    The metrics are the frobenius differences between the cvariance calculated with the covariance_calculator and the
    optimal or true covariance
    :param dataset_covariance: the dataset covariance, a dxd psd matrix
    :param covariance_calculator: a CovarianceMatrixCalculator instance
    :param permute_target_matrix_diagonal: whether to permute the target matrix diagonal to get a smaller distance
    (needed when the dataset_covariance is proportional to identity)
    :param gan: a GANwithMultipleDiscriminators instance, the model to be used
    :param latent_dim: the dimension of the generator input
    :param SGAN_model: if True, the optimal covariance will be calculated for an SGAN, otherwise an entropic W2GAN
    solution will be used
    :param epsilon: the entropic regularization parameter, only needed if SGAN_model is False
    :return: a dictionary of the metrics
    """
    if SGAN_model:
        optimal_covariance = get_optimal_output_covariance_gaussian_sgan(dataset_covariance, latent_dim)
    else:
        optimal_covariance = get_optimal_output_covariance_gaussian_ew2gan(dataset_covariance, latent_dim, epsilon)

    def batch_covariance_distance_to_optimal_cov(generator_output, real_data,
                                                 generated_data_discriminator_list_outputs,
                                                 real_data_discriminator_list_outputs):
        return torch.tensor(frobenius_distance_to_generator_output_covariance(
            optimal_covariance,
            generator_output,
            covariance_calculator,
            gan,
            permute_target_matrix_diagonal=permute_target_matrix_diagonal,
            use_sample_covariance=True
        ))

    def batch_covariance_distance_to_true_cov(generator_output, real_data,
                                              generated_data_discriminator_list_outputs,
                                              real_data_discriminator_list_outputs):
        return torch.tensor(frobenius_distance_to_generator_output_covariance(
            dataset_covariance,
            generator_output,
            covariance_calculator,
            gan,
            permute_target_matrix_diagonal=permute_target_matrix_diagonal,
            use_sample_covariance=True
        ))

    def running_covariance_distance_to_optimal_cov(generator_output, real_data,
                                                   generated_data_discriminator_list_outputs,
                                                   real_data_discriminator_list_outputs):
        return torch.tensor(frobenius_distance_to_generator_output_covariance(
            optimal_covariance,
            generator_output,
            covariance_calculator,
            gan,
            permute_target_matrix_diagonal=permute_target_matrix_diagonal,
            use_sample_covariance=False
        ))

    def running_covariance_distance_to_true_cov(generator_output, real_data,
                                                generated_data_discriminator_list_outputs,
                                                real_data_discriminator_list_outputs):
        return torch.tensor(frobenius_distance_to_generator_output_covariance(
            dataset_covariance,
            generator_output,
            covariance_calculator,
            gan,
            permute_target_matrix_diagonal=permute_target_matrix_diagonal,
            use_sample_covariance=False
        ))

    return {'batch covariance distance to true cov': batch_covariance_distance_to_true_cov,
            'batch covariance distance to optimal cov': batch_covariance_distance_to_optimal_cov,
            'running covariance distance to true cov': running_covariance_distance_to_true_cov,
            'running covariance distance to optimal cov': running_covariance_distance_to_optimal_cov}