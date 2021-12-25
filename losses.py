import sys
import os
import torch
import numpy as np
import scipy
import scipy.linalg
# I changed the geomloss library!!!! (samples_loss.py and sinkhorn_divergence.py) to return
# both dual potentials
import geomloss

sys.path.append(os.getcwd())


def squared_pairwise_distances(x, y=None):
    """
    Function to calculate the pairwise distances matrix
    :param x: an Nxd matrix
    :param y: an optional Mxd matirix, if y is not given then use 'y=x'
    :return: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    """
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    # torch.square(torch.squeeze(torch.cdist(x.view((1,)+x.shape), y.view((1,)+x.shape), p=2)))
    return dist


def compute_gradient_penalty_for_wgan(discriminator, real_data, generated_data):
    """
    Computes the gradient penalty for the discriminator of the Wassersten-1 GAN. Since the gradient norm along the
    lines connecting fake and real data points has norm 1, the condition is enforced on random points from the lines
    :param discriminator: the discriminator, which should be Lipschitz
    :param real_data: the real data sample
    :param generated_data: the generated data (generator output on random noise)
    :return: the gradient penalty loss -- the squared difference between the norm of the discriminator (input is
    randomly sampled to be on the line connecting true and generated data points) and 1, averaged over samples
    """

    # Get the batch size from real data
    batch_size = real_data.shape[0]
    # Choose a point on the line connecting real_data and generated_data.
    # First, sample the distance to the real_data point for every sample
    alpha = torch.rand(batch_size, *([1]*(real_data.ndim - 1))).to(real_data.device)
    # Second, construct the points on the line
    sampled_points = (alpha * real_data + (1 - alpha) * generated_data)
    sampled_points.requires_grad_(True)

    # Get discriminator score for the sampled points
    sampled_points_scores = discriminator(sampled_points)
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=torch.sum(sampled_points_scores),
        inputs=sampled_points,
        create_graph=True
    )[0]

    # Compute the gradient norm
    gradient_norm = gradients.view(batch_size, -1).norm(2, 1)
    return torch.mean((gradient_norm - 1) ** 2)


def wasserstein_loss_from_potentials(x_potential, y_potential):
    """
    Calculate the 1-Wasserstein loss between uniform measures supported on x and y using their potentials assumes that
    the first argument is the real sample and the second one is the fake sample potentials (other way around also works,
    the order is for consistency with other methods)
    :param x_potential: the potential value on the variable x
    :param y_potential: the potential value on the variable y
    :return: 1-Wasserstein loss calculated from the potentials, assuming
    """
    return x_potential.mean() - y_potential.mean()


def sinkhorn_loss(x, y, epsilon, return_potentials=False):
    """
    Calculate the Sinkhorn loss between uniform measures supported on x and y with regularization parameter epsilon. If
    return_potentials is true, return the dual potentials of the Sinkhorn loss. Note that this code uses geomloss library
    https://www.kernel-operations.io/geomloss/api/pytorch-api.html, but the calculation of Sinkhorn loss was modified to
    return the dual potentials. Also, the Sinkhorn loss defined here is different from the way it is defined in the
    geomloss library: loss_here(x, y, epsilon) = 2 * geomloss_loss(x, y, sqrt(epsilon/2))
    :param x: support of the first measure
    :param y:  support of the second measure
    :param epsilon: regularization strength(blur parameter in the geomloss library)
    :param return_potentials: boolean, whether to output the dual potentials or the Sinkhorn loss value
    :return: Sinkhorn loss if return_potentials is False and dual potentials of the Sinkhorn loss otherwise
    """
    if not return_potentials:
        return 2 * geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=np.sqrt(epsilon / 2),
                                        debias=True)(x, y)
    f_x, f_y = geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=np.sqrt(epsilon / 2),
                                    debias=True, potentials=True)(x, y)
    halved_potentials = list(f_x + f_y)
    potentials = [2 * f for f in halved_potentials]
    return potentials


def entropic_2Wasserstein_loss(x, y, epsilon, return_potentials=False):
    """
    Calculate the entropy-regularized Wasserstein distance between uniform measures supported on x and y with
    regularization parameter epsilon. If
    return_potentials is true, return the dual potentials of the Sinkhorn loss. Note that this code uses geomloss library
    https://www.kernel-operations.io/geomloss/api/pytorch-api.html, Also, the loss defined here is different from the
    way it is defined in the geomloss library: loss_here(x, y, epsilon) = 2 * geomloss_loss(x, y, sqrt(epsilon/2))
    :param x: support of the first measure
    :param y:  support of the second measure
    :param epsilon: regularization strength(blur parameter in the geomloss library)
    :param return_potentials: boolean, whether to output the dual potentials or the Sinkhorn loss value
    :return: entropy-regularized Wasserstein distance if return_potentials is False and dual potentials of the
    loss otherwise
    """
    if not return_potentials:
        return 2 * geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=np.sqrt(epsilon / 2),
                                        debias=False)(x, y)
    f_x, f_y = geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=np.sqrt(epsilon / 2),
                                    debias=False, potentials=True)(x, y)
    halved_potentials = [f_x, f_y]
    potentials = [2 * f for f in halved_potentials]
    return potentials


def entropic_2Wasserstein_loss_dual_from_potentials(x, y, epsilon, x_potential, y_potential):
    """
    Calculate the Sinkhorn loss between uniform measures supported on x and y when the dual potentials are known using
    the dual formulation. Note that for the optimal dual potentials loss = E[x_potential] + E[y_potential], but
    here we include the term epsilon*E[e^((x_potential ⊕ y_potential - squared_distance_xy)/epsilon) - 1] to account for
    the non-optimal dual potentials
    :param x: support of the first measure
    :param y: support of the second measure
    :param epsilon: regularization strength(blur parameter in the geomloss library)
    :param x_potential: the potential corresponding to x
    :param y_potential: the potential corresponding to y
    """
    squared_distance_xy = squared_pairwise_distances(x, y)
    to_exponentiate = (x_potential.view(-1, 1) + y_potential.view(1, -1) - squared_distance_xy) / epsilon
    max_to_exponentiate = torch.max(to_exponentiate)
    loss = x_potential.mean() + y_potential.mean() - epsilon * torch.mean(torch.exp(to_exponentiate -
                                                                                    max_to_exponentiate)) * \
           torch.exp(max_to_exponentiate) + epsilon
    # print("Exponent argument maximum is {:.3e}".format(float(max_to_exponentiate.data)))
    return loss


def entropic_2Wasserstein_loss_semidual_from_potential(x, y, epsilon, x_potential):
    """
    Calculate the entropy-regularized 2-Wasserstein loss between uniform measures supported on x and y when the dual
    potential for x is known using the semi-dual formulation (https://arxiv.org/pdf/1811.05527.pdf):
    loss = E[x_potential] - epsilon * E[log(E[exp((x_potential - \|x - y\|_2^2)/epsilon)])]
    :param x: support of the first measure
    :param y: support of the second measure
    :param epsilon: regularization strength(blur parameter in the geomloss library)
    :param x_potential: the potential corresponding to x
    """
    squared_distance_xy = squared_pairwise_distances(x, y)
    to_exponentiate = (x_potential.view(-1, 1) - squared_distance_xy) / epsilon - np.log(x.shape[0])
    loss = x_potential.mean() - epsilon * torch.logsumexp(to_exponentiate, dim=0).mean()
    return loss


def get_optimal_second_potential_for_entropic_2Wasserstein_loss(x, y, epsilon, x_potential):
    """
    Calculate the optimal dual potential of y for a given dual potential of x for the Entropy-regularized 2-Wasserstein
    distance between uniform measures supported on x and y
    :param x: support of the first measure
    :param y: support of the second measure
    :param epsilon: regularization strength(blur parameter in the geomloss library)
    :param x_potential: the potential corresponding to x
    :return: the potential corresponding to y
    """
    squared_distance_xy = squared_pairwise_distances(x, y)
    to_exponentiate = (x_potential.view(-1, 1) - squared_distance_xy) / epsilon - np.log(x.shape[0])
    potential_y = - epsilon * torch.logsumexp(to_exponentiate, dim=0)
    return potential_y


def entropic_2Wasserstein_loss_primal_from_potentials(x, y, epsilon, x_potential, y_potential,
                                                      normalize_probability=False, normalize_distances=False):
    """
    Calculate the Sinkhorn loss between uniform measures supported on x and y when the dual potentials are known using
    the primal formulation, i.e. computing the probability first and then treating it as a constant (This will be used
    for generator gradients)
    :param x: support of the first measure
    :param y: support of the second measure
    :param epsilon: regularization strength(blur parameter in the geomloss library)
    :param x_potential: the potential corresponding to x
    :param y_potential: the potential corresponding to y
    :param normalize_probability: boolean, whether to normalize the calculated probability matrix to sum to 1
    :param normalize_distances: boolean, whether to normalize the squared distances between x and y to have a maximum
                                    of 1 during calculations (this does not change the answer, but may reduce numerical
                                     errors)
    :return loss: Sinkhorn loss
    """
    squared_distance_xy = squared_pairwise_distances(x, y)

    log_probability_xy = (x_potential.view(-1, 1) + y_potential.view(1, -1) - squared_distance_xy) / epsilon - \
                         np.log(x.shape[0] * y.shape[0])
    if normalize_probability:
        log_probability_xy -= torch.logsumexp(log_probability_xy, dim=(0, 1))
    log_probability_xy = log_probability_xy.detach()
    probability_xy = torch.exp(log_probability_xy).detach()

    # Uncomment to print the probability matrix sums -- to check how close it is to a real coupling
    # print("Probability: sum {:.3e}, marginal x mismatch: {:.3e}, "
    #       "marginal y mismatch: {:.3e}".format(torch.sum(probability_xy).data,
    #                                           *[torch.norm(torch.sum(probability_xy, dim=1-i) \
    #                                                      - torch.ones(probability_xy.shape[i]) \
    #                                                      / probability_xy.shape[i]).data
    #                                                      for i in range(2)]))
    mutual_information = torch.sum(probability_xy * log_probability_xy) + np.log(y.shape[0]) + np.log(x.shape[0])
    if normalize_distances:
        max_dist = torch.max(squared_distance_xy)
        normalized_squared_distance_xy = squared_distance_xy / max_dist
        expected_squared_distance = torch.sum(normalized_squared_distance_xy * probability_xy) * max_dist
    else:
        expected_squared_distance = torch.sum(squared_distance_xy * probability_xy)
    loss = expected_squared_distance + epsilon * mutual_information
    # print("Expected squared distance {:.3e}, MI: {:.3e}".format(expected_squared_distance,mutual_information))
    return loss


def sinkhorn_loss_dual_from_potentials(x, y, epsilon, x_potential, xx_potential, y_potential, yy_potential=None):
    """
    Calculate the Sinkhorn loss between uniform measures supported on x and y when the dual potentials are known using
    the dual formulation
    :param x: support of the first measure
    :param y: support of the second measure
    :param epsilon: regularization strength(blur parameter in the geomloss library)
    :param x_potential: the potential corresponding to x in W_2^2(P_x, P_y)
    :param y_potential: the potential corresponding to y in W_2^2(P_x, P_y)
    :param xx_potential: the potential corresponding to x in W_2^2(P_x, P_x)
    :param yy_potential: the potential corresponding to x in W_2^2(P_y, P_y), if None W_2^2(P_y, P_y) is not calculated
                        and not included in the Sinkhorn loss)
    """
    # x_potential -= x_potential.mean()
    loss_xy = entropic_2Wasserstein_loss_dual_from_potentials(x, y, epsilon, x_potential, y_potential)
    loss_xx = entropic_2Wasserstein_loss_dual_from_potentials(x, x, epsilon, xx_potential, xx_potential)
    if yy_potential is not None:
        loss_yy = entropic_2Wasserstein_loss_dual_from_potentials(y, y, epsilon, yy_potential, yy_potential)
    else:
        loss_yy = 0
    return loss_xy - 0.5 * (loss_xx + loss_yy)


def sinkhorn_loss_primal_from_potentials(x, y, epsilon, x_potential, xx_potential, y_potential, yy_potential=None,
                                         normalize_probability=False, normalize_distances=False):
    """
    Calculate the Sinkhorn loss between uniform measures supported on x and y when the dual potentials are known using
    the dual formulation
    :param x: support of the first measure
    :param y: support of the second measure
    :param epsilon: regularization strength(blur parameter in the geomloss library)
    :param x_potential: the potential corresponding to x in W_2^2(P_x, P_y)
    :param y_potential: the potential corresponding to y in W_2^2(P_x, P_y)
    :param xx_potential: the potential corresponding to x in W_2^2(P_x, P_x)
    :param yy_potential: the potential corresponding to x in W_2^2(P_y, P_y), if None W_2^2(P_y, P_y) is not calculated
                        and not included in the Sinkhorn loss)
    :param normalize_probability: boolean, whether to normalize the calculated probability matrix to sum to 1 in the
                                    entropic W2 distances
    :param normalize_distances: boolean, whether to normalize the squared distances between x and y to have a maximum
                                    of 1 during calculations in the entropic W2 distances (this does not change the
                                    answer, but may reduce numerical errors but also influence gradients)
    """
    loss_xy = entropic_2Wasserstein_loss_primal_from_potentials(x, y, epsilon, x_potential, y_potential,
                                                                normalize_probability=normalize_probability,
                                                                normalize_distances=normalize_distances)
    loss_xx = entropic_2Wasserstein_loss_primal_from_potentials(x, x, epsilon, xx_potential, xx_potential,
                                                                normalize_probability=normalize_probability,
                                                                normalize_distances=normalize_distances)
    if yy_potential is not None:
        loss_yy = entropic_2Wasserstein_loss_primal_from_potentials(y, y, epsilon, yy_potential, yy_potential,
                                                                    normalize_probability=normalize_probability,
                                                                    normalize_distances=normalize_distances)
    else:
        loss_yy = 0
    return loss_xy - 0.5 * (loss_xx + loss_yy)


def sinkhorn_loss_semidual_from_x_potential(x, y, epsilon, x_potential, xx_potential=None, yy_potential=None):
    """
    Calculate the Sinkhorn loss between uniform measures supported on x and y when the one of the dual potentials is
    known using the semi-dual formulation
    :param x: support of the first measure
    :param y: support of the second measure
    :param epsilon: regularization strength(blur parameter in the geomloss library)
    :param x_potential: the potential corresponding to x in W_2^2(P_x, P_y)
    :param xx_potential: the potential corresponding to x in W_2^2(P_x, P_x), if None W_2^2(P_y, P_y) is not calculated
                        and not included in the Sinkhorn loss), but only one of xx_potential or yy_potential can be
                        None, if both are None an error is thrown
    :param yy_potential: the potential corresponding to x in W_2^2(P_y, P_y), if None W_2^2(P_y, P_y) is not calculated
                        and not included in the Sinkhorn loss)
    """
    # x_potential -= x_potential.mean()
    if xx_potential is None and yy_potential is None:
        raise ValueError("Both dual potentials for the debiasing term are None. Consider using entropy-regularized "
                         "Wasserstein distance in this case or specify one of the debiasing term dual potentials")
    loss_xy = entropic_2Wasserstein_loss_semidual_from_potential(x, y, epsilon, x_potential)
    if xx_potential is not None:
        loss_xx = entropic_2Wasserstein_loss_dual_from_potentials(x, x, epsilon, xx_potential, xx_potential)
    else:
        loss_xx = 0
    if yy_potential is not None:
        loss_yy = entropic_2Wasserstein_loss_dual_from_potentials(y, y, epsilon, yy_potential, yy_potential)
    else:
        loss_yy = 0
    return loss_xy - 0.5 * (loss_xx + loss_yy)


def true_entropic_2Wasserstein_loss_gaussian(mean_x, mean_y, cov_x, cov_y, epsilon):
    """
    Calculate the entropy-regularized 2-Wasserstein distance between 2 multivariate Gaussians
    :param mean_x: mean of the variable x (torch.Tensor or a numpy array)
    :param mean_y: mean of the variable y (torch.Tensor or a numpy array)
    :param cov_x:  covariance matrix of the variable x (torch.Tensor or a numpy array)
    :param cov_y:  covariance matrix of the variable y (torch.Tensor or a numpy array)
    :param epsilon: regularization strength
    :return: Entropy-regularized 2-Wasserstein distance with regularization strength epsilon assuming that both input
    distributions are multivariate Gaussian with means and covariances provided as parameters
    """
    dim = cov_x.shape[0]
    difference_in_means = np.sum(np.square(mean_x - mean_y))
    sigma_squared = epsilon / 2
    cov_x_rt = scipy.linalg.sqrtm(cov_x)
    D_sigma = scipy.linalg.sqrtm(4 * cov_x_rt.dot(cov_y).dot(cov_x_rt.T) +
                                 (sigma_squared ** 2) * np.eye(dim))
    loss = np.trace(cov_x) + np.trace(cov_y) - np.trace(D_sigma) + dim * sigma_squared * \
           (1 - np.log(2 * sigma_squared)) + sigma_squared * np.log(np.linalg.det(D_sigma + sigma_squared * np.eye(dim)))
    loss += difference_in_means
    return loss


def quadratic_function_of_array_or_tensor(matrix, bias, x):
    """
    This function is a helper function that returns x^T*matrix * x + bias in case x is a vector, otherwise
    it assumes that x is N x d with each row representing a vector x_i, then the function returns a vector of
    x_i^T*matrix*x_i + bias
    :param matrix: (numpy array or torch.Tensor) the matrix of the bilinear form x^T*matrix*x+bias
    :param bias: (numpy array or torch.Tensor or int) the bias of the bilinear form x^T*matrix*x+bias
    :param x:(numpy array or torch.Tensor) the argument
    :return: x^T*matrix * x + bias in case x is a vector, otherwise returns a vector of with ith component being
    x_i^T*matrix*x_i + bias
    """
    # assume that x is either a 1-dim vector or a matrix N x d
    if type(x) == torch.Tensor:
        matrix = torch.tensor(matrix)
        if x.is_cuda:
            matrix = matrix.cuda()
        return (torch.matmul(x, matrix) * x).sum(1) + bias
    else:
        return np.sum(x.dot(matrix) * x, axis=-1) + bias


def true_entropic_2Wasserstein_loss_dual_potentials_gaussian(mean_x, mean_y, cov_x, cov_y, epsilon):
    """
    Calculate true dual potentials for Entropy-regularized 2-Wasserstein distance between two Gaussians
    :param mean_x: mean of the variable x (torch.Tensor or a numpy array)
    :param mean_y: mean of the variable y (torch.Tensor or a numpy array)
    :param cov_x:  covariance matrix of the variable x (torch.Tensor or a numpy array)
    :param cov_y:  covariance matrix of the variable y (torch.Tensor or a numpy array)
    :param epsilon: regularization strength
    :return: a tuple of functions, the dual potentials for Entropy-regularized 2-Wasserstein distance between two
    Gaussians
    """
    dim = cov_x.shape[0]
    difference_in_means = np.sum(np.square(mean_x - mean_y))

    # Convert epsilon to sigma^2 for consistency with https://arxiv.org/pdf/2006.02572.pdf
    sigma_squared = epsilon / 2

    # Formula (15) from https://arxiv.org/pdf/2006.02572.pdf
    cov_x_rt = scipy.linalg.sqrtm(cov_x)
    D_sigma = scipy.linalg.sqrtm(4 * cov_x_rt.dot(cov_y).dot(cov_x_rt.T) +
                                 (sigma_squared ** 2) * np.eye(dim))
    C_sigma = cov_x_rt.dot(D_sigma).dot(np.linalg.inv(cov_x_rt))/2 - sigma_squared / 2 * np.eye(dim)
    # Formula (23) from https://arxiv.org/pdf/2006.02572.pdf times sigma_squared
    U_times_sigma_squared = (cov_y.dot(np.linalg.inv(C_sigma + sigma_squared * np.eye(dim))) - np.eye(dim))
    V_times_sigma_squared = (np.linalg.inv(C_sigma + sigma_squared * np.eye(dim)).dot(cov_x) - np.eye(dim))

    # Add dimension to the mean in case it's a vector for proper broadcasting
    dimension_adder = lambda x, y: y[np.newaxis, :] if (x.ndim == 2 and type(y) == np.ndarray) else y

    # Calculate the bias for the dual potentials (they are defined up to a constant, since only the bias of
    # their sum matters, so the biases are equal for both potentials)
    # Easiest way to calculate -- use (11) from https://arxiv.org/pdf/2006.02572.pdf with x = y = 0
    bias = 0.5 * (sigma_squared * np.log(np.linalg.det(C_sigma / sigma_squared + np.eye(dim))) + difference_in_means)

    # Formula (19) from https://arxiv.org/pdf/2006.02572.pdf for scaled U and V.
    # Note that Q(A) = -1/2 * x^TAx (see comment to formula 16)
    f = lambda x: quadratic_function_of_array_or_tensor(-U_times_sigma_squared, bias, x - dimension_adder(x, mean_x))
    g = lambda y: quadratic_function_of_array_or_tensor(-V_times_sigma_squared, bias, y - dimension_adder(y, mean_y))
    return f, g
