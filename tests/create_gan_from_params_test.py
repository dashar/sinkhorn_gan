import utils
import losses
import data_processing
import torch
import numpy as np
import architecture

# Set the parameters
sample_size = 1000
data_dim = 32
latent_dim = 4
use_random_covariance = False
discriminator_parameter_name = 'main.dense_0.weight'
generator_parameter_name = 'main.dense_0.weight'
gradient_penalty = 100
batch_size = 1000
entropic_regularization_strength = 12.0

# Create the dataset with a fixed covariance and generate a sample of latent data and real data
dataset = data_processing.GaussianDataset(sample_size, data_dim, latent_dim, use_random_covariance=False)
latent_data = torch.tensor(np.random.randn(batch_size, latent_dim)).float()
real_data = torch.tensor(dataset.X[:batch_size, :]).float()

# Create expected losses dictionaries
discriminator_losses_dict = {
    "WGAN": lambda gan, latent_data, real_data: -losses.wasserstein_loss_from_potentials(
        gan.discriminator_list[0](gan.generator(latent_data)),
        gan.discriminator_list[0](real_data)
    ),
    "WGAN-GP": lambda gan, latent_data, real_data: (-losses.wasserstein_loss_from_potentials(
        gan.discriminator_list[0](gan.generator(latent_data)),
        gan.discriminator_list[0](real_data)
    ) + gradient_penalty * losses.compute_gradient_penalty_for_wgan(gan.discriminator_list[0], real_data,
                                                                    gan.generator(latent_data))),
    "SGAN-NP": lambda gan, latent_data, real_data: None,
    "SGAN-P": lambda gan, latent_data, real_data: losses.sinkhorn_loss_dual_from_potentials(
        gan.generator(latent_data), real_data, entropic_regularization_strength,
        gan.generated_data_discriminator_list[0](gan.generator(latent_data)),
        gan.generated_data_discriminator_list[1](gan.generator(latent_data)),
        gan.real_data_discriminator_list[0](real_data)
    ),
    "EW2GAN": lambda gan, latent_data, real_data: losses.entropic_2Wasserstein_loss_dual_from_potentials(
        gan.generator(latent_data), real_data, entropic_regularization_strength,
        gan.generated_data_discriminator_list[0](gan.generator(latent_data)),
        gan.real_data_discriminator_list[0](real_data)
    )
}
generator_losses_dict = {
    "WGAN": lambda gan, latent_data, real_data: losses.wasserstein_loss_from_potentials(
        gan.discriminator_list[0](gan.generator(latent_data)),
        gan.discriminator_list[0](real_data)
    ),
    "WGAN-GP": lambda gan, latent_data, real_data: losses.wasserstein_loss_from_potentials(
        gan.discriminator_list[0](gan.generator(latent_data)),
        gan.discriminator_list[0](real_data)
    ),
    "SGAN-NP": lambda gan, latent_data, real_data: losses.sinkhorn_loss(gan.generator(latent_data), real_data,
                                                                        entropic_regularization_strength),
    "SGAN-P": lambda gan, latent_data, real_data: losses.sinkhorn_loss_primal_from_potentials(
        gan.generator(latent_data), real_data, entropic_regularization_strength,
        gan.generated_data_discriminator_list[0](gan.generator(latent_data)),
        gan.generated_data_discriminator_list[1](gan.generator(latent_data)),
        gan.real_data_discriminator_list[0](real_data)
    ),
    "EW2GAN": lambda gan, latent_data, real_data: losses.entropic_2Wasserstein_loss_primal_from_potentials(
        gan.generator(latent_data), real_data, entropic_regularization_strength,
        gan.generated_data_discriminator_list[0](gan.generator(latent_data)),
        gan.real_data_discriminator_list[0](real_data)
    )
}

# Create a GAN
for GAN_loss_and_structure in ['WGAN', 'WGAN-GP', 'SGAN-NP', 'SGAN-P', 'EW2GAN']:
    print("\n" + GAN_loss_and_structure + "\n")
    gan, generator_loss, discriminator_loss = architecture.create_gan_from_params(GAN_loss_and_structure if GAN_loss_and_structure != 'WGAN-GP' else 'WGAN',
                                                                               data_dimension=data_dim,
                                                                               latent_dimension=latent_dim,
                                                                               GAN_n_hidden_layer_neurons=64,
                                                                               use_linear_generator=True,
                                                                               entropic_regularization_strength=entropic_regularization_strength,
                                                                               gradient_penalty_strength=0 if '-GP' not in
                                                                                                              GAN_loss_and_structure else gradient_penalty,
                                                                               use_cuda=torch.cuda.is_available())

    # Save the generator weight
    generator_weight = gan.generator.get_parameter(generator_parameter_name).clone().detach().numpy()

    # Calculate the losses and their gradients for the sampled data
    d_optimizer = torch.optim.SGD(sum([list(discriminator.parameters())
                                       for discriminator in gan.discriminator_list], []), lr=0.0) \
        if len(gan.discriminator_list)>0 else None
    g_optimizer = torch.optim.SGD(gan.generator.parameters(), lr=0.0)

    # compile the GAN with the aforementioned losses and optimizers
    gan.compile(discriminator_loss, generator_loss, d_optimizer, g_optimizer)

    # Define the discriminator_parameter and generator_parameter wrt which the gradient will be calculated --
    # the first linear layer weight
    discriminator_parameter = gan.generated_data_discriminator_list[0].get_parameter(discriminator_parameter_name) if len(gan.discriminator_list)>0 else None
    generator_parameter = gan.generator.get_parameter(generator_parameter_name)

    for phase, expected_loss, train_step, gradient_parameter in zip(
        ["discriminator", "generator"],
        [discriminator_losses_dict[GAN_loss_and_structure], generator_losses_dict[GAN_loss_and_structure]],
        [gan.discriminator_train_step, gan.generator_train_step],
        [discriminator_parameter, generator_parameter]
    ):
        if phase == 'discriminator' and GAN_loss_and_structure == 'SGAN-NP':
            continue
        # calculate the loss and its gradient calculated at train step
        train_step_loss = train_step(latent_data, real_data)['{} loss'.format(phase)]
        calculated_gradient = gradient_parameter.grad.clone().detach().numpy()

        # calculate the expected loss and its gradient
        expected_loss_tensor = expected_loss(gan, latent_data, real_data)
        # print("expected loss: ", expected_loss_tensor)
        # print('discriminator parameter name', discriminator_parameter_name)
        expected_gradient = torch.autograd.grad(expected_loss_tensor, gradient_parameter)[0].clone().detach().numpy()

        # Calculate the differences in losses and gradients
        expected_discriminator_loss = float(expected_loss_tensor.detach().numpy())
        difference_expected_and_calculated_loss = np.abs(expected_discriminator_loss - train_step_loss)
        difference_expected_and_calculated_gradient = np.linalg.norm(expected_gradient - calculated_gradient)


        # Print the difference
        print("difference between expected and calculated {} loss: {:.3e}".format(
            phase,
            difference_expected_and_calculated_loss
        ))
        print("difference between expected and calculated gradients of the {} loss: {:.3e}".format(
            phase,
            difference_expected_and_calculated_gradient
        ))

    # Check that the gradients are the same
    # TODO: check that in Sinkhorn and entropic Gans the mean of one of the discriminators is substracted and added to the
    #  other (since the mean is irrelevant)