import torch

from architecture import *
from data_processing import *
from losses import *
import numpy as np
from torch.utils import tensorboard
from metrics import *
import torchvision

# create a dataset with a random covariance
dataset = GaussianDataset(sample_size=1000, data_dim=32, latent_dim=4, use_random_covariance=True)
real_data_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
# create a GAN model -- we are testing WGAN
G = Generator()
D = Discriminator()
GAN = GANwithMultipleDiscriminators(G, [D], [D])

# create loss functions
loss_d = lambda x, y, x_potentials_list, y_potentials_list: -wasserstein_loss_from_potentials(x_potentials_list[0],
                                                                                              y_potentials_list[0])
loss_g = lambda x, y, x_potentials_list, y_potentials_list: wasserstein_loss_from_potentials(x_potentials_list[0],
                                                                                             y_potentials_list[0])

# use sgd optimizers with no momentum and a large learning rate to see the updated values fast
d_optimizer = torch.optim.SGD(D.parameters(), lr=0.001)
g_optimizer = torch.optim.SGD(G.parameters(), lr=0.001)



# compile the GAN with the aforementioned losses and optimizers
GAN.compile(loss_d, loss_g, d_optimizer, g_optimizer)


# latent_data = torch.tensor(np.random.randn(10, 4)).float()
# real_data = torch.tensor(dataset.X[:10, :]).float()

# create a function to sample from the latent distribution (1-d Gaussian)
def latent_data_fn(batch_size):
    return torch.Tensor(np.random.randn(batch_size, 4)).float()

# sample a batch of real data and latent distribution
real_data = next(iter(real_data_loader))
latent_data = latent_data_fn(real_data.shape[0])

print(real_data.shape)
print(latent_data.shape)

# function to print the means of weights of the first layer of the networks
def calculate_first_layer_weight_means(GAN):
    # discriminators:
    discriminator_first_layer_weight_means = [float(dict(D.named_parameters())['main.dense_0.weight'].mean().detach().numpy())
                                              for D in GAN.discriminator_list]
    generator_first_layer_weight_mean = float(dict(GAN.generator.named_parameters())['main.dense_0.weight'].mean().detach()
                                              .numpy())
    return discriminator_first_layer_weight_means, generator_first_layer_weight_mean


print('############## DISCRIMINATOR STEP ##############')
# calculate the NN layer 0 weight means to compare to the ones after optimization step
discriminator_first_layer_weight_means_before_disc_step, generator_first_layer_weight_mean_before_disc_step = \
    calculate_first_layer_weight_means(GAN)

# calculate the expected discriminator loss and the one calculated at discriminator train step
expected_discriminator_loss = -float((D(G(latent_data)).mean() - D(real_data).mean()).detach().numpy())
train_step_discriminator_loss = GAN.discriminator_train_step(latent_data, real_data)['discriminator loss']
difference_expected_and_calculated_discriminator_loss = np.abs(expected_discriminator_loss -
                                                               train_step_discriminator_loss)
print("difference between expected and calculated discriminator loss: {:.3e}".format(
    difference_expected_and_calculated_discriminator_loss))

discriminator_first_layer_weight_means_after_disc_step, generator_first_layer_weight_mean_after_disc_step = \
    calculate_first_layer_weight_means(GAN)

print("NN parameters changes:")
print(("avg. discriminator weight change: " + ("{:.3e}, "*len(GAN.discriminator_list))[:-2]).format(
    *list(np.array(discriminator_first_layer_weight_means_before_disc_step)
                  - np.array(discriminator_first_layer_weight_means_after_disc_step))
))

print("avg. generator weight change: {:.3e}".format(generator_first_layer_weight_mean_before_disc_step -
                                                   generator_first_layer_weight_mean_after_disc_step))

print('############## GENERATOR STEP ##############')
# calculate the NN layer 0 weight means to compare to the ones after optimization step
discriminator_first_layer_weight_means_before_gen_step, generator_first_layer_weight_mean_before_gen_step = \
    discriminator_first_layer_weight_means_after_disc_step, generator_first_layer_weight_mean_after_disc_step


# calculate the expected discriminator loss and the one calculated at generator train step
expected_generator_loss = float((D(G(latent_data)).mean() - D(real_data).mean()).detach().numpy())
train_step_generator_loss = GAN.generator_train_step(latent_data, real_data)['generator loss']
difference_expected_and_calculated_generator_loss = np.abs(expected_generator_loss - train_step_generator_loss)
print("difference between expected and calculated discriminator loss: {:.3e}".format(
    difference_expected_and_calculated_generator_loss))

discriminator_first_layer_weight_means_after_gen_step, generator_first_layer_weight_mean_after_gen_step = \
    calculate_first_layer_weight_means(GAN)

print("NN parameters changes:")
print(("avg. discriminator weight change: " + ("{:.3e}, "*len(GAN.discriminator_list))[:-2]).format(
    *list(np.array(discriminator_first_layer_weight_means_before_gen_step)
                  - np.array(discriminator_first_layer_weight_means_after_gen_step))
))

print("avg. generator weight change: {:.3e}".format(generator_first_layer_weight_mean_before_gen_step -
                                                    generator_first_layer_weight_mean_after_gen_step))

print(len(list(real_data_loader)))
writer = tensorboard.SummaryWriter("tensorboard_logs")
out = GAN.train(latent_data_fn, real_data_loader, 10, functions_to_run_at_epoch_start=[],
              use_new_real_data_for_generator_step=True, use_new_latent_data_for_generator_step=False,
              discriminator_iterations_per_generator_iterations=5, validation_real_data_loader=real_data_loader,
              tensorboard_writer=writer, n_generator_iterations_to_print_batch_metrics=100)
writer.flush()
writer.close()
cumulative_generator_step_metrics, cumulative_discriminator_step_metrics, epoch_averaged_discriminator_step_metrics, \
    epoch_averaged_generator_step_metrics, epoch_validation_step_metrics = out
