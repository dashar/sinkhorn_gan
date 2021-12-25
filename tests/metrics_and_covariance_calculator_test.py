import torch

from metrics import *
from architecture import Generator, Discriminator, GANwithMultipleDiscriminators
from  data_processing import generate_random_covariance

# Test the CovarianceMatrixCalculator first

# Create a sample of N(0,I) 32-dimansional data points
data_dim = 32
dataset_size = 1000
sample = np.random.randn(dataset_size, data_dim)
sample_mean = np.mean(sample, axis=0)
data_covariance = np.cov(sample.T)

# Initialize the covariance calculator and update it
print("COVARIANCE CALCULATOR TEST")
covariance_calculator = CovarianceMatrixCalculator()
covariance_calculator_estimate = covariance_calculator.update(torch.tensor(sample))
print("Frobenius dist. between sample covariance and covariance_calculator estimate: {:.3e}\n"
      .format(np.linalg.norm(covariance_calculator_estimate - data_covariance)))

covariance_calculator_estimate = covariance_calculator.update(torch.tensor(10 * sample))
print("After an update with updated=True and 10*sample (has to not have any effect):")
print("\tFrobenius dist. between sample covariance and covariance_calculator estimate: {:.3e}"
      .format(np.linalg.norm(covariance_calculator_estimate - data_covariance)))

covariance_calculator.set_updated_to_false()
covariance_calculator_estimate = covariance_calculator.update(torch.tensor(10 * sample))
print("After an update with updated=False and 10*sample (has to change the difference to be large):")
print("\tFrobenius dist. between sample covariance and covariance_calculator estimate: {:.3e}"
      .format(np.linalg.norm(covariance_calculator_estimate - data_covariance)))
expected_covariance = np.cov(np.concatenate([sample, 10*sample], axis=0).T)
print("\tFrobenius dist. between the now-expected covariance and covariance_calculator estimate: {:.3e}"
      .format(np.linalg.norm(covariance_calculator_estimate - expected_covariance)))

print("Test a smaller batch:")
covariance_calculator.set_updated_to_false()
small_sample = 200 * sample[:2, :]
covariance_calculator_estimate = covariance_calculator.update(torch.tensor(small_sample))
expected_covariance = np.cov(np.concatenate([sample, 10*sample, small_sample], axis=0).T)
print("\tFrobenius dist. between the now-expected covariance and covariance_calculator estimate: {:.3e}"
      .format(np.linalg.norm(covariance_calculator_estimate - expected_covariance)))

covariance_calculator.reset()
covariance_calculator_estimate = covariance_calculator.update(torch.tensor(sample))
print("After reset:")
print("\tFrobenius dist. between sample covariance and covariance_calculator estimate: {:.3e}\n"
      .format(np.linalg.norm(covariance_calculator_estimate - data_covariance)))

print("\n\nfrobenius_distance_to_generator_output_covariance TEST")
print("Linear GAN")
latent_dim = 4
hidden_layer_size = 64
G = Generator(n_inputs=latent_dim, n_neurons=hidden_layer_size, data_dim=data_dim, linear=True)
D = Discriminator(n_neurons=hidden_layer_size, data_dim=data_dim)
GAN = GANwithMultipleDiscriminators(G, [D], [D])
generator_weight_before_metric_calculation = G.get_parameter('main.dense_0.weight').detach().numpy()

# Set the parameters
target_matrix = np.eye(data_dim) / np.sqrt(data_dim)
covariance_calculator.reset()
generator_output = torch.tensor(sample)

calculated_distance = frobenius_distance_to_generator_output_covariance(target_matrix, generator_output,
                                                                        covariance_calculator, GAN,
                                                                        permute_target_matrix_diagonal=False,
                                                                        use_sample_covariance=False)
expected_distance = np.linalg.norm(generator_weight_before_metric_calculation.dot(
    generator_weight_before_metric_calculation.T) - target_matrix)
print("\tCheck that the Generator parameters didn't change: the difference is {:.3e}".format(np.linalg.norm(
    generator_weight_before_metric_calculation - G.get_parameter('main.dense_0.weight').detach().numpy())))
print("\tCheck that the distance was calculated correctly: the difference is {:.3e}".format(
    np.linalg.norm(calculated_distance - expected_distance)))
print("\tCheck that the covariance_calculator was updated: the difference with true covariance is {:.3e}".format(
    np.linalg.norm(covariance_calculator.covariance - data_covariance)))

# Now we check that sample covariance works and for that we will change the
# covariance_calculator.covariance to see the change (if it happens)
covariance_calculator.covariance *= 2
calculated_distance = frobenius_distance_to_generator_output_covariance(target_matrix, generator_output,
                                                                        covariance_calculator, GAN,
                                                                        permute_target_matrix_diagonal=False,
                                                                        use_sample_covariance=True)
expected_distance = np.linalg.norm(np.cov(sample.T) - target_matrix)
print("\tCheck that the distance was calculated correctly with use_sample_covariance=True: the difference is {:.3e}"
      .format(np.linalg.norm(calculated_distance - expected_distance)))

# Now we check that permute_target_matrix_diagonal=True works. For that we have to change the target matrix to have
# different diagonal values and we also change the covariance matrix of the covariance_calculator to have the same
# values on the diagonal, but permuted. We expect the distance to be  0.
target_matrix = np.diag(np.concatenate([np.arange(5), np.zeros(data_dim-5)]))
pad = (0, 28)
G.get_parameter('main.dense_0.weight').data = torch.tensor(np.pad(np.diag(np.sqrt(np.arange(4, 0, -1))), [pad, (0, 0)]))
calculated_distance = frobenius_distance_to_generator_output_covariance(target_matrix, generator_output,
                                                                        covariance_calculator, GAN,
                                                                        permute_target_matrix_diagonal=True,
                                                                        use_sample_covariance=False)
print("\tCheck that the distance was calculated correctly with permute_target_matrix_diagonal=True: "
      "the difference is {:.3e}".format(calculated_distance))

print("\nNon-linear GAN")
latent_dim = 4
hidden_layer_size = 64
G = Generator(n_inputs=latent_dim, n_neurons=hidden_layer_size, data_dim=data_dim)
GAN = GANwithMultipleDiscriminators(G, [D], [D])
generator_weight_before_metric_calculation = G.get_parameter('main.dense_0.weight').detach().numpy()

# Set the parameters
target_matrix = np.eye(data_dim) / np.sqrt(data_dim)
covariance_calculator.reset()
generator_output = torch.tensor(sample)
calculated_distance = frobenius_distance_to_generator_output_covariance(target_matrix, generator_output,
                                                                        covariance_calculator, GAN,
                                                                        permute_target_matrix_diagonal=False,
                                                                        use_sample_covariance=False)
expected_distance = np.linalg.norm(np.cov(sample.T) - target_matrix)
print("\tCheck that the Generator parameters didn't change: the difference is {:.3e}".format(np.linalg.norm(
    generator_weight_before_metric_calculation - G.get_parameter('main.dense_0.weight').detach().numpy())))
print("\tCheck that the distance was calculated correctly: the difference is {:.3e}".format(
    np.linalg.norm(calculated_distance - expected_distance)))
print("\tCheck that the covariance_calculator was updated: the difference with true covariance is {:.3e}".format(
    np.linalg.norm(covariance_calculator.covariance - data_covariance)))

# Now we check that sample covariance works and for a non-liner GAN we will change the
# covariance_calculator.covariance to see the change (if it happens)
covariance_calculator.covariance *= 2
calculated_distance = frobenius_distance_to_generator_output_covariance(target_matrix, generator_output,
                                                                        covariance_calculator, GAN,
                                                                        permute_target_matrix_diagonal=False,
                                                                        use_sample_covariance=True)
expected_distance = np.linalg.norm(np.cov(sample.T) - target_matrix)
print("\tCheck that the distance was calculated correctly with use_sample_covariance=True: the difference is {:.3e}"
      .format(np.linalg.norm(calculated_distance - expected_distance)))
# Now we check that permute_target_matrix_diagonal=True works. For that we have to change the target matrix to have
# different diagonal values and we also change the covariance matrix of the covariance_calculator to have the same
# values on the diagonal, but permuted. We expect the distance to be  0.
target_matrix = np.diag(np.concatenate([np.arange(5), np.zeros(data_dim-5)]))
covariance_calculator.covariance = np.diag(np.concatenate([np.zeros(data_dim-5), np.arange(4, -1, -1)]))
calculated_distance = frobenius_distance_to_generator_output_covariance(target_matrix, generator_output,
                                                                        covariance_calculator, GAN,
                                                                        permute_target_matrix_diagonal=True,
                                                                        use_sample_covariance=False)
print("\tCheck that the distance was calculated correctly with permute_target_matrix_diagonal=True: "
      "the difference is {:.3e}".format(calculated_distance))

print("\n\nOPTIMAL COVARIANCE CALCULATOR TEST")
print("SGAN")
# We have to check that the largest latent_dim eigenvectors are the same and all the others are 0
# First generate a random psd matrix and get its eigenvalues and eigenvectors (note that np.linalg.eigh returns them in
# ascending (↑) order)
true_covariance = generate_random_covariance(data_dim)
true_eigenvals, true_eigenvecs = np.linalg.eigh(true_covariance)
optimal_covariance = get_optimal_output_covariance_gaussian_sgan(true_covariance, latent_dim)
optimal_eigenvals, optimal_eigenvecs = np.linalg.eigh(optimal_covariance)
print("\tLargest latent_dim eigenvalues difference: {:.3e}".format(np.linalg.norm(true_eigenvals[-latent_dim:] -
                                                                                optimal_eigenvals[-latent_dim:])))
print("\tOther eigenvalues norm: {:.3e}".format(np.linalg.norm(optimal_eigenvals[:-latent_dim])))
print("\tChecking that the first latent_dim eigenvectors coincide: {:.3e}".format(
    np.linalg.norm(optimal_eigenvecs[:, -latent_dim:] * np.sign(true_eigenvecs[:1, -latent_dim:])
                   * np.sign(optimal_eigenvecs[:1, -latent_dim:]) - true_eigenvecs[:, -latent_dim:])
))
print("\nENTROPIC W2GAN")
# We have to check that the largest latent_dim eigenvectors of the optimal covariance are positive parts of the largest
# latent_dim eigenvectors of the true covariance minus lambda/2  and all the other eigenvalues are 0
# First generate a random psd matrix and get its eigenvalues and eigenvectors (note that np.linalg.eigh returns them in
# ascending (↑) order)
true_covariance = generate_random_covariance(data_dim)
true_eigenvals, true_eigenvecs = np.linalg.eigh(true_covariance)
regularization_strength = 0.1
optimal_covariance = get_optimal_output_covariance_gaussian_entropic_wgan(true_covariance, latent_dim,
                                                                          regularization_strength)
optimal_eigenvals, optimal_eigenvecs = np.linalg.eigh(optimal_covariance)
print("\tLargest latent_dim eigenvalues difference: {:.3e}".format(np.linalg.norm(
    np.maximum(true_eigenvals[-latent_dim:] - regularization_strength/2, 0) - optimal_eigenvals[-latent_dim:])))
print("\tOther eigenvalues norm: {:.3e}".format(np.linalg.norm(optimal_eigenvals[:-latent_dim])))
print("\tChecking that the first latent_dim eigenvectors coincide: {:.3e}".format(
    np.linalg.norm(optimal_eigenvecs[:, -latent_dim:] * np.sign(true_eigenvecs[:1, -latent_dim:])
                   * np.sign(optimal_eigenvecs[:1, -latent_dim:]) - true_eigenvecs[:, -latent_dim:])
))