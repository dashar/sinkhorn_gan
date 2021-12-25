import os, sys

sys.path.append(os.getcwd())

import torch
import numpy as np
import losses
import collections


def print_dictionary(name, dictionary, index=-1):
    print((name + ("{}: {:.5e}, " * len(dictionary))[:-1]).format(*[i if type(i) not in [list, np.ndarray] else
                                                                    float(i) if type(i) == np.ndarray and i.ndim == 0
                                                                    else i[index]
                                                                  for pair in dictionary.items() for i in pair]))
    # print("checking print_dictioonary function, args:", name, dictionary)

class Generator(torch.nn.Module):
    """
    class of the Generator neural networks, a linear function or a 3 hidden layer neural network
    (input -> hidden layer 1 (linear+ReLu) -> hidden layer 2 (linear+ReLu) -> hidden layer 3 (linear+ReLu) ->
    output(linear))
    """

    def __init__(self, n_inputs=4, n_neurons=64, data_dim=32, linear=False):
        """
        Generator Neural Network initialization
        :param n_inputs: number of input neurons, the latent space dimension (the dimension of the input to the
                            Generator)
        :param n_neurons: number of hidden layer neurons (noy used when linear is True), 64 by default
        :param data_dim: the dimension of the generator output, 32 by default
        :param linear: (boolean) if True, the generator is linear
        """
        super(Generator, self).__init__()
        if linear:
            main = torch.nn.Sequential(
                collections.OrderedDict([
                    ('dense_0', torch.nn.Linear(n_inputs, data_dim))
                ])
            )
        else:
            main = torch.nn.Sequential(
                collections.OrderedDict([
                    ('dense_0', torch.nn.Linear(n_inputs, n_neurons)),
                    ('relu_0', torch.nn.ReLU(True)),
                    ('dense_1', torch.nn.Linear(n_neurons, n_neurons)),
                    ('relu_1', torch.nn.ReLU(True)),
                    ('dense_2', torch.nn.Linear(n_neurons, n_neurons)),
                    ('relu_2', torch.nn.ReLU(True)),
                    ('dense_3', torch.nn.Linear(n_neurons, data_dim)),
                ])
            )
        self.main = main
        self.linear = linear

    def forward(self, noise):
        output = self.main(noise)
        return output


class Discriminator(torch.nn.Module):
    """
    class of the Discriminator neural networks, a 3 hidden layer neural network
    (input -> hidden layer 1 (linear+ReLu) -> hidden layer 2 (linear+ReLu) -> hidden layer 3 (linear+ReLu) ->
    output(linear))
    """

    def __init__(self, n_neurons=64, data_dim=32):
        """
        Discriminator neural network initialization (3 hidden layers with ReLu activations)
        :param n_neurons: number of neurons in the hidden layers
        :param data_dim: data dimension (the dimension of the input to the discriminator since
                    the output dimension is 1)
        """
        super(Discriminator, self).__init__()
        main = torch.nn.Sequential(
            collections.OrderedDict([
                ('dense_0', torch.nn.Linear(data_dim, n_neurons)),
                ('relu_0', torch.nn.ReLU(True)),
                ('dense_1', torch.nn.Linear(n_neurons, n_neurons)),
                ('relu_1', torch.nn.ReLU(True)),
                ('dense_2', torch.nn.Linear(n_neurons, n_neurons)),
                ('relu_2', torch.nn.ReLU(True)),
                ('dense_3', torch.nn.Linear(n_neurons, 1)),
            ])
        )
        self.main = main

    def forward(self, inputs):
        output = self.main(inputs)
        return output.view(-1)


class GANwithMultipleDiscriminators:
    """
    A class used to represent GANs with multiple discriminators, taking generator output or real data as input or both

    Attributes
    ----------
    generator : Generator
        the generator used in the GAN
    real_data_discriminator_list : list of Discriminator / Neural Networks
        the discriminators that use real data as input
    generated_data_discriminator_list : list of Discriminator / Neural Networks
        the discriminators that use generated data as input (the list can have a non-empty intersection with the
        real_data_discriminator_list)
    discriminator_list : list of Discriminator / Neural Networks
        a list of all the discriminators used in the model (the union of real_data_discriminator_list and
        generated_data_discriminator_list)
    use_cuda : boolean
        if True, the networks are put on cuda and gradient calculations are done on cuda
    TODO: complete the list
    generator_optimizer : None
    self.discriminator_optimizer = None
    self.clip_value = None
    self.generator_step_metrics = None
    self.discriminator_step_metrics = None
    self.validation_metrics = None
    self.discriminator_loss = None
    self.generator_loss = None
    self.discriminators_to_clip_weights = None
    says_str : str
        a formatted string to print out what the animal says
    name : str
        the name of the animal
    sound : str
        the sound that the animal makes
    num_legs : int
        the number of legs the animal has (default 4)

    Methods
    -------
    #TODO: list the methods
    """

    def __init__(self, generator, real_data_discriminator_list, generated_data_discriminator_list, use_cuda=False):
        self.generator = generator
        self.real_data_discriminator_list = real_data_discriminator_list
        self.generated_data_discriminator_list = generated_data_discriminator_list

        # Create a list of all the discriminators (both the ones for real data and the ones for generated data)
        self.discriminator_list = self.real_data_discriminator_list
        for discriminator in self.generated_data_discriminator_list:
            if discriminator not in self.discriminator_list:
                self.discriminator_list.append(discriminator)

        # Put the networks on cuda if the parameter was specified
        self.use_cuda = use_cuda
        if use_cuda:
            self.generator.cuda()
            for discriminator in self.discriminator_list:
                discriminator.cuda()

        # Set the yet unknown parameters to None
        self.generator_optimizer = None
        self.discriminator_optimizer = None
        self.clip_value = None
        self.generator_step_metrics = None
        self.discriminator_step_metrics = None
        self.validation_metrics = None
        self.discriminator_loss = None
        self.generator_loss = None
        self.discriminators_to_clip_weights = None

    def __call__(self, generator_input):
        return self.generator(generator_input)

    def compile(self, discriminator_loss, generator_loss, discriminator_optimizer, generator_optimizer,
                discriminator_step_metrics=None, generator_step_metrics=None, validation_metrics=None,
                metric_names_not_to_average=[], discriminators_to_clip_weights=None, clip_value=(-.01, .01)):
        """
        Compiles the model by setting the optimizers, losses and metrics for discriminator and generator
        :param discriminator_loss: function that takes generator output, real_data,  generated_data_discriminator_list
                                    outputs and real_data_discriminator_list outputs as input and
                                    outputs a single number -- the discriminator loss.
        :param generator_loss: function that takes generator output, real_data, generated_data_discriminator_list
                                    outputs and real_data_discriminator_list as input and outputs a
                                    single number -- the generator loss
        :param generator_optimizer: optimizer for the generator loss
        :param discriminator_optimizer: optimizer for the discriminator loss
        :param discriminator_step_metrics: dictionary of metrics that will be output after the discriminator step,
                                            'discriminator loss' is added to the metrics if it is not there
        :param generator_step_metrics: dictionary of metrics that will be output after the generator step,
                                            'generator loss' is added to the metrics if it is not there
        :param validation_metrics: dictionary of metrics that will be calculated at a validation step,
                                            'generator loss' and discriminator losses are added to the metrics if not
                                            there
        :param metric_names_not_to_average: a list of metric names that need not be averaged, the last value will be
        passed to the epochwise metrics
        :param discriminators_to_clip_weights: the list of discriminators that need their weights clipped to
                                                clip_value
        :param clip_value: tuple of 2 numbers (in increasing order), weights of discriminators_to_clip_weights will be
                            clipped to this interval
        :return: void
        """
        # Set the losses as model attributes
        self.discriminator_loss = discriminator_loss
        self.generator_loss = generator_loss

        # Set the discriminators that will need their weights clipped to empty
        # if discriminators_to_clip_weights is None:
        if discriminators_to_clip_weights is None:
            self.discriminators_to_clip_weights = []
        else:
            self.discriminators_to_clip_weights = discriminators_to_clip_weights
        self.clip_value = clip_value

        # Set the metrics to be an empty dictionary if it is None
        generator_step_metrics = {} if generator_step_metrics is None else generator_step_metrics
        discriminator_step_metrics = {} if discriminator_step_metrics is None else discriminator_step_metrics
        validation_metrics = {} if validation_metrics is None else validation_metrics

        # Add losses to the metrics if they are not already there
        if 'generator loss' not in generator_step_metrics:
            generator_step_metrics['generator loss'] = self.generator_loss
        if 'discriminator loss' not in discriminator_step_metrics and len(self.discriminator_list) > 0:
            discriminator_step_metrics['discriminator loss'] = self.discriminator_loss
        if 'generator loss' not in validation_metrics:
            validation_metrics['generator loss'] = self.generator_loss
        if 'discriminator loss' not in validation_metrics and len(self.discriminator_list) > 0:
            validation_metrics['discriminator loss'] = self.discriminator_loss

        # Set the parameters of the GAN
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_step_metrics = generator_step_metrics
        self.discriminator_step_metrics = discriminator_step_metrics
        self.validation_metrics = validation_metrics
        self.metric_names_not_to_average = metric_names_not_to_average

    def generator_train_step(self, generator_input, real_data=None):
        # Set the generator gradients to 0 and put the generator input and real data on cuda
        self.generator.zero_grad()
        if self.use_cuda:
            generator_input = generator_input.cuda()
            if real_data is not None:
                real_data = real_data.cuda()

        # Get the generated data
        generator_output = self.generator(generator_input)

        # Calculate the real data discriminators outputs (set to None if real_data is None)
        if real_data is not None:
            real_data_discriminator_outputs = [discriminator(real_data).detach() for discriminator in
                                               self.real_data_discriminator_list]
        else:
            real_data_discriminator_outputs = None

        # Calculate the generated data discriminators outputs
        generated_data_discriminator_outputs = [discriminator(generator_output) if real_data is not None else None
                                                for discriminator in self.generated_data_discriminator_list]

        # Get the loss value and do the gradient update
        generator_loss_value = self.generator_loss(generator_output, real_data, generated_data_discriminator_outputs,
                                                   real_data_discriminator_outputs)
        # print("Generated data discriminator output", generated_data_discriminator_outputs)
        # print("Real data discriminator output", real_data_discriminator_outputs)
        # print("Generator output", generator_output)
        generator_loss_value.backward()
        self.generator_optimizer.step()

        # Calculate the metrics
        metrics_values = {key: metric_fn(generator_output, real_data, generated_data_discriminator_outputs,
                                         real_data_discriminator_outputs).cpu().detach().numpy()
                          for key, metric_fn in self.generator_step_metrics.items()}
        return metrics_values

    def discriminator_train_step(self, generator_input, real_data):
        # Handle the case of Sinkhorn GAN when there are no discriminators and therefore no discriminator steps
        if len(self.discriminator_list) == 0:
            return {}
        # Set the gradients to 0 and put the generator_input data and real data on cuda
        for discriminator in self.discriminator_list:
            discriminator.zero_grad()
        if self.use_cuda:
            generator_input = generator_input.cuda()
            real_data = real_data.cuda()

        # Get the generated data and detach it since no pass will be done on the generator
        generator_output = self.generator(generator_input).detach()

        # Calculate the real data discriminators outputs
        real_data_discriminator_outputs = [discriminator(real_data)
                                           for discriminator in self.real_data_discriminator_list]

        # Calculate the generated data discriminators outputs
        generated_data_discriminator_outputs = [discriminator(generator_output)
                                                for discriminator in self.generated_data_discriminator_list]

        # Calculate the forward pass, backward pass and do the gradient updates
        discriminator_loss_value = self.discriminator_loss(generator_output, real_data,
                                                           generated_data_discriminator_outputs,
                                                           real_data_discriminator_outputs)
        discriminator_loss_value.backward()
        self.discriminator_optimizer.step()

        # Clip weights of discriminator
        if self.clip_value is not None:
            for discriminator in self.discriminators_to_clip_weights:
                for param in discriminator.parameters():
                    param.data.clamp_(self.clip_value[0], self.clip_value[1])

        # Calculate the metrics (essentially this is the metrics from the last step's updates)
        metrics_values = {key: metric_fn(generator_output, real_data, generated_data_discriminator_outputs,
                                         real_data_discriminator_outputs).cpu().detach().numpy()
                          for key, metric_fn in self.discriminator_step_metrics.items()}
        return metrics_values

    def calculate_validation_metrics(self, get_new_generator_input_fn, real_data_loader):
        averaged_metrics = {key: 0.0 for key in self.validation_metrics.keys()}
        n_batches_seen = 0
        # with torch.no_grad():
        for real_data in real_data_loader:
            generator_input = get_new_generator_input_fn(real_data.shape[0])

            if self.use_cuda:
                generator_input = generator_input.cuda()
                real_data = real_data.cuda()
            # Get the networks outputs
            generator_output = self.generator(generator_input)
            real_data_discriminator_outputs = [discriminator(real_data)
                                               for discriminator in self.real_data_discriminator_list]
            generated_data_discriminator_outputs = [discriminator(generator_output)
                                                    for discriminator in self.generated_data_discriminator_list]
            for metric_name, metric_fn in self.validation_metrics.items():
                metric_value = metric_fn(generator_output, real_data, generated_data_discriminator_outputs,
                                         real_data_discriminator_outputs).cpu().detach().numpy()

                if metric_name not in self.metric_names_not_to_average:
                    averaged_metrics[metric_name] = (averaged_metrics[metric_name] * n_batches_seen + metric_value) / \
                                                    (n_batches_seen + 1)
                else:
                    averaged_metrics[metric_name] = metric_value
            n_batches_seen += 1
        return averaged_metrics

    def train(self, get_new_generator_input_fn, real_data_loader, epochs, functions_to_run_at_epoch_start=[],
              functions_to_run_at_generator_step_start=[], use_new_real_data_for_generator_step=True,
              use_new_latent_data_for_generator_step=True, discriminator_iterations_per_generator_iterations=5,
              validation_real_data_loader=None, tensorboard_writer=None,
              n_generator_iterations_to_print_batch_metrics=None):
        """
        :param get_new_generator_input_fn: function that takes the batch_size and outputs a random sample of the
                                                generator input (this is not an iterator since we are expecting a
                                                random sample every iteration and we want the batch size of the latent
                                                variable to match the batch size of real data) -- MAYBE SHOULD BE
                                                ITERATOR
        :param real_data_loader: torch.utils.data.DataLoader class or any iterative  that contains real data batches
        :param epochs: number of epochs (passes through the entire real_data dataset) to train for
        :param functions_to_run_at_epoch_start: list of functions that will be run at the beginning of every epoch
                                                (might be needed for correct metric calculations), empty list by default
        :param functions_to_run_at_generator_step_start: list of functions that will be run at the beginning of every
                                                generator step (might be needed for correct metric calculations),
                                                empty list by default
        :param use_new_real_data_for_generator_step: (boolean) if true, the batch of real data used for generator step
                                                        is different from the batch used in the latest discriminator
                                                        step
        :param use_new_latent_data_for_generator_step: (boolean) if true, the batch of latent data used for generator
                                                        step is different from the batch used in the latest
                                                        discriminator step
        :param discriminator_iterations_per_generator_iterations: number of discriminator gradient steps per each
                                                                    generator gradient step
                                                                    (not in use if there is no discriminator)
        :param validation_real_data_loader: torch.utils.data.DataLoader class or any iterative that contains real data
                                            batches to be used for validation at the end of every epoch
        :return:
        """

        # initialize the epoch use_averaged_covariance metrics by zeros
        if len(self.discriminator_list) > 0:
            epoch_averaged_discriminator_step_metrics = {key: np.zeros(epochs)
                                                         for key in self.discriminator_step_metrics}
        else:
            epoch_averaged_discriminator_step_metrics = {}
        epoch_averaged_generator_step_metrics = {key: np.zeros(epochs) for key in self.generator_step_metrics}

        if validation_real_data_loader is not None:
            epoch_validation_step_metrics = {key: np.zeros(epochs) for key in self.validation_metrics}
        else:
            epoch_validation_step_metrics = {}

        # set the global step iterators
        global_discriminator_iteration = 0
        global_generator_iteration = 0

        # initialize the iterationwise metrics (all the metrics history)
        if len(self.discriminator_list) > 0:
            iterationwise_discriminator_step_metrics = {key: [] for key in self.discriminator_step_metrics}
        else:
            iterationwise_discriminator_step_metrics = {}
        iterationwise_generator_step_metrics = {key: [] for key in self.generator_step_metrics}

        for epoch in range(epochs):
            for fn in functions_to_run_at_epoch_start:
                fn()
            discriminator_iteration = 0
            generator_iteration = 0
            in_generator_iteration = False

            # set line length for progress printing
            line_length = 40
            epoch_start_string = " EPOCH {:4d} starting "
            print("#" * line_length)
            print("-" * ((line_length - len(epoch_start_string)) // 2) + epoch_start_string.format(epoch) +
                  "-" * ((line_length - len(epoch_start_string) + 1) // 2))
            print("#" * line_length)

            for real_data in real_data_loader:
                # Make a discriminator step if we are not in the generator step (the generator step is needed when
                # new real data batch is drawn for every generator step, i.e. use_new_real_data_for_generator_step
                # is True)
                if (not in_generator_iteration) and len(self.discriminator_list) > 0:
                    generator_input = get_new_generator_input_fn(real_data.shape[0])
                    current_discriminator_step_metrics = self.discriminator_train_step(generator_input, real_data)
                    # print("discriminator step {}, generator input mean: {}, real_data mean: {}, metrics: ".format(
                    #     global_discriminator_iteration, generator_input.cpu().detach().numpy().mean(),
                    #     real_data.cpu().detach().numpy().mean()),
                    #       current_discriminator_step_metrics)
                    for key, value in current_discriminator_step_metrics.items():
                        iterationwise_discriminator_step_metrics[key].append(value)
                        if key not in self.metric_names_not_to_average:
                            epoch_averaged_discriminator_step_metrics[key][epoch] *= \
                                discriminator_iteration / (1 + discriminator_iteration)
                            epoch_averaged_discriminator_step_metrics[key][epoch] += value / (
                                        1 + discriminator_iteration)
                        else:
                            epoch_averaged_discriminator_step_metrics[key][epoch] = value

                        # if 'discriminator' not in key:
                        #     key = 'discriminator_' + key
                        if tensorboard_writer is not None:
                            tensorboard_writer.add_scalar('per_iteration_' + key.replace(' ', '_') + '/train', value,
                                                          global_discriminator_iteration)
                    global_discriminator_iteration += 1
                    discriminator_iteration += 1
                    in_generator_iteration = (global_discriminator_iteration
                                              % discriminator_iterations_per_generator_iterations == 0)
                    if in_generator_iteration and use_new_real_data_for_generator_step:
                        continue

                # Make a generator step
                if in_generator_iteration or len(self.discriminator_list) == 0:
                    for fn in functions_to_run_at_generator_step_start:
                        fn()
                    if use_new_latent_data_for_generator_step:
                        generator_input = get_new_generator_input_fn(real_data.shape[0])
                    current_generator_step_metrics = self.generator_train_step(generator_input, real_data)
                    # print("generator step {}, generator input mean: {}, real_data mean: {}, metrics: ".format(
                    #     global_generator_iteration, generator_input.cpu().detach().numpy().mean(),
                    #     real_data.cpu().detach().numpy().mean()),
                    #       current_generator_step_metrics)
                    for key, value in current_generator_step_metrics.items():
                        iterationwise_generator_step_metrics[key].append(value)
                        if key not in self.metric_names_not_to_average:
                            epoch_averaged_generator_step_metrics[key][epoch] *= \
                                generator_iteration / (1 + generator_iteration)
                            epoch_averaged_generator_step_metrics[key][epoch] += value / (1 + generator_iteration)
                        else:
                            epoch_averaged_generator_step_metrics[key][epoch] = value
                        # if 'generator' not in key:
                        #     key = 'generator_' + key
                        if tensorboard_writer is not None:
                            tensorboard_writer.add_scalar('per_iteration_' + key.replace(' ', '_') + '/train', value,
                                                          global_generator_iteration)
                    global_generator_iteration += 1
                    generator_iteration += 1
                    in_generator_iteration = False
                    if (n_generator_iterations_to_print_batch_metrics is not None) and \
                            (global_generator_iteration % n_generator_iterations_to_print_batch_metrics == 0):
                        print("Generator iteration {:5d}:".format(global_generator_iteration))
                        print_dictionary("Generator metrics: ", epoch_averaged_generator_step_metrics, index=epoch)
                        if len(self.discriminator_list) > 0:
                            print_dictionary("Discriminator metrics: ", epoch_averaged_discriminator_step_metrics,
                                             index=epoch)

            # print final epoch metrics
            print("EPOCH {:4d} ended".format(epoch))
            print_dictionary("Generator metrics: ", epoch_averaged_generator_step_metrics, index=epoch)
            if len(self.discriminator_list) > 0:
                print_dictionary("Discriminator metrics: ", epoch_averaged_discriminator_step_metrics, index=epoch)

            if validation_real_data_loader is not None:
                # print("Evaluating validation metrics")
                validation_metrics = self.calculate_validation_metrics(get_new_generator_input_fn,
                                                                       validation_real_data_loader)
                print_dictionary("Validation metrics: ", validation_metrics)
                for key, value in validation_metrics.items():
                    epoch_validation_step_metrics[key][epoch] = value
                    if tensorboard_writer is not None:
                        tensorboard_writer.add_scalar(key.replace(' ', '_') + '/val', value,
                                                      epoch)
            if tensorboard_writer is None:
                continue
            # Write all data to tensorboard if the writer is provided
            # Write epochwise metrics to tensorboard
            for key, value in epoch_averaged_generator_step_metrics.items():
                tensorboard_writer.add_scalar(key.replace(' ', '_') + '/train', value[epoch], epoch)
            for key, value in epoch_averaged_discriminator_step_metrics.items():
                tensorboard_writer.add_scalar(key.replace(' ', '_') + '/train', value[epoch], epoch)
            # Write weight histograms and images of the generator to tensorboard
            for name, param in self.generator.named_parameters():
                if 'weight' in name or 'bias' in name:
                    legible_name = name.split('.')[1:]
                    legible_name = 'generator_layer_' + '_'.join(legible_name)
                    if 'weight' in name:
                        weight = param.clone().cpu().detach().numpy()
                        tensorboard_writer.add_histogram(legible_name, weight.flatten(), epoch)
                        weight -= np.min(weight)
                        weight /= np.max(weight)
                        tensorboard_writer.add_image(legible_name + '_img', weight, epoch,
                                                     dataformats='HW')
                    else:
                        tensorboard_writer.add_histogram(legible_name, param.cpu().detach().numpy(), epoch)

            # Write weight histograms and images of the discriminators to tensorboard
            n_real_data_discriminators_seen = 0
            n_generated_data_discriminators_seen = 0
            n_both_data_discriminators_seen = 0
            for discriminator in self.discriminator_list:
                if discriminator in self.real_data_discriminator_list:
                    if discriminator not in self.generated_data_discriminator_list:
                        n_real_data_discriminators_seen += 1
                        name_prefix = 'generated_data_discriminator_{}_'.format(n_real_data_discriminators_seen)
                    else:
                        n_both_data_discriminators_seen += 1
                        name_prefix = 'both_data_discriminator_{}_'.format(n_both_data_discriminators_seen)
                else:
                    n_generated_data_discriminators_seen += 1
                    name_prefix = 'both_data_discriminator_{}_'.format(n_generated_data_discriminators_seen)

                for name, param in discriminator.named_parameters():
                    if 'weight' in name or 'bias' in name:
                        legible_name = name.split('.')[1:]
                        legible_name = name_prefix + 'layer_' + '_'.join(legible_name)
                        if 'weight' in name:
                            weight = param.clone().cpu().detach().numpy()
                            tensorboard_writer.add_histogram(legible_name, weight.flatten(), epoch)
                            weight -= np.min(weight)
                            weight /= np.max(weight)
                            tensorboard_writer.add_image(legible_name+'_img',  weight, epoch,dataformats='HW')
                        else:
                            tensorboard_writer.add_histogram(legible_name, param.cpu().detach().numpy(), epoch)

        return [iterationwise_generator_step_metrics, iterationwise_discriminator_step_metrics,
                epoch_averaged_discriminator_step_metrics, epoch_averaged_generator_step_metrics,
                epoch_validation_step_metrics]


def create_gan_from_params(GAN_loss_and_structure, data_dimension=32, latent_dimension=4, GAN_n_hidden_layer_neurons=64,
                           use_linear_generator=False, entropic_regularization_strength=12.0,
                           gradient_penalty_strength=0, use_cuda=torch.cuda.is_available()):

    # First, check if the if gradient_penalty_strength > 0 only for a wasserstein-1 gan (for other gan architectures
    # the gradient_penalty_strength != 0 doesn't make sense since the lipschitzness is not necessary)
    if gradient_penalty_strength > 0 and GAN_loss_and_structure != 'WGAN':
        raise ValueError("gradient_penalty_strength > 0 is only allowed with GAN_loss_and_structure = WGAN")

    # Create the model and the loss functions for the Generator and the Discriminator. Since every GAN model requires a
    # generator, create it first
    generator = Generator(n_inputs=latent_dimension, n_neurons=GAN_n_hidden_layer_neurons,
                                       data_dim=data_dimension, linear=use_linear_generator)
    # Now create the discriminators (if needed) and define the loss functions
    if GAN_loss_and_structure == 'SGAN-NP':
        gan = GANwithMultipleDiscriminators(generator=generator, real_data_discriminator_list=[],
                                                         generated_data_discriminator_list=[], use_cuda=use_cuda)

        def generator_loss(generator_output, real_data, generated_data_discriminator_list_outputs,
                           real_data_discriminator_list_outputs):
            return losses.sinkhorn_loss(generator_output, real_data, entropic_regularization_strength,
                                        return_potentials=False)

        discriminator_loss = None

    elif GAN_loss_and_structure == 'SGAN-P':
        real_data_discriminator = Discriminator(n_neurons=GAN_n_hidden_layer_neurons,
                                                             data_dim=data_dimension)

        generated_data_discriminator_wassersterin_term = Discriminator(
            n_neurons=GAN_n_hidden_layer_neurons,
            data_dim=data_dimension)
        generated_data_discriminator_debiasing_term = Discriminator(
            n_neurons=GAN_n_hidden_layer_neurons,
            data_dim=data_dimension)

        gan = GANwithMultipleDiscriminators(generator=generator,
                                                         real_data_discriminator_list=[real_data_discriminator],
                                                         generated_data_discriminator_list=[
                                                             generated_data_discriminator_wassersterin_term,
                                                             generated_data_discriminator_debiasing_term
                                                         ],
                                                         use_cuda=use_cuda)

        def generator_loss(generator_output, real_data, generated_data_discriminator_list_outputs,
                           real_data_discriminator_list_outputs):
            return losses.sinkhorn_loss_primal_from_potentials(
                x=generator_output,
                y=real_data,
                epsilon=entropic_regularization_strength,
                x_potential=generated_data_discriminator_list_outputs[0],
                xx_potential=generated_data_discriminator_list_outputs[1],
                y_potential=real_data_discriminator_list_outputs[0],
                yy_potential=None,
                normalize_probability=False,
                normalize_distances=False
            )

        def discriminator_loss(generator_output, real_data, generated_data_discriminator_list_outputs,
                               real_data_discriminator_list_outputs):
            # Note that in the case of SGAN-P the discriminators have to optimize (maximize) the entropic 2-Wasserstein
            # distances between real and generated data and between generated data and generated data, so we use
            # negative of  their sum as a loss function
            loss_xy = -losses.entropic_2Wasserstein_loss_dual_from_potentials(
                x=generator_output,
                y=real_data,
                epsilon=entropic_regularization_strength,
                x_potential=generated_data_discriminator_list_outputs[0],
                y_potential=real_data_discriminator_list_outputs[0]
            )
            loss_xx = -losses.entropic_2Wasserstein_loss_dual_from_potentials(
                x=generator_output,
                y=generator_output,
                epsilon=entropic_regularization_strength,
                x_potential=generated_data_discriminator_list_outputs[1],
                y_potential=generated_data_discriminator_list_outputs[1]
            )
            return loss_xy + loss_xx
    # semi-dual SGAN with a discriminator corresponding to real data
    elif GAN_loss_and_structure == 'SGAN-RealSD':
        real_data_discriminator = Discriminator(n_neurons=GAN_n_hidden_layer_neurons, data_dim=data_dimension)
        generated_data_discriminator_debiasing_term = Discriminator(
            n_neurons=GAN_n_hidden_layer_neurons,
            data_dim=data_dimension)

        gan = GANwithMultipleDiscriminators(generator=generator,
                                                         real_data_discriminator_list=[real_data_discriminator],
                                                         generated_data_discriminator_list=[
                                                             generated_data_discriminator_debiasing_term
                                                         ],
                                                         use_cuda=use_cuda)

        def generator_loss(generator_output, real_data, generated_data_discriminator_list_outputs,
                           real_data_discriminator_list_outputs):
            return losses.sinkhorn_loss_semidual_from_x_potential(
                x=real_data,
                y=generator_output,
                epsilon=entropic_regularization_strength,
                x_potential=real_data_discriminator_list_outputs[0],
                yy_potential=generated_data_discriminator_list_outputs[0]
            )

        def discriminator_loss(generator_output, real_data, generated_data_discriminator_list_outputs,
                               real_data_discriminator_list_outputs):
            # Note that in the case of SGAN-P the discriminators have to optimize (maximize) the entropic 2-Wasserstein
            # distances between real and generated data and between generated data and generated data, so we use
            # negative of  their sum as a loss function
            loss_xy = -losses.entropic_2Wasserstein_loss_semidual_from_potential(
                x=real_data,
                y=generator_output,
                epsilon=entropic_regularization_strength,
                x_potential=real_data_discriminator_list_outputs[0]
            )
            loss_xx = -losses.entropic_2Wasserstein_loss_semidual_from_potential(
                x=generator_output,
                y=generator_output,
                epsilon=entropic_regularization_strength,
                x_potential=generated_data_discriminator_list_outputs[0]
            )
            return loss_xy + loss_xx
        # semi-dual SGAN with a discriminator corresponding to real data
    elif GAN_loss_and_structure == 'SGAN-FakeSD':
        generated_data_discriminator = Discriminator(n_neurons=GAN_n_hidden_layer_neurons, data_dim=data_dimension)
        generated_data_discriminator_debiasing_term = Discriminator(
            n_neurons=GAN_n_hidden_layer_neurons,
            data_dim=data_dimension)

        gan = GANwithMultipleDiscriminators(generator=generator,
                                            real_data_discriminator_list=[],
                                            generated_data_discriminator_list=[
                                                generated_data_discriminator,
                                                generated_data_discriminator_debiasing_term
                                            ],
                                            use_cuda=use_cuda)

        def generator_loss(generator_output, real_data, generated_data_discriminator_list_outputs,
                           real_data_discriminator_list_outputs):
            return losses.sinkhorn_loss_semidual_from_x_potential(
                x=generator_output,
                y=real_data,
                epsilon=entropic_regularization_strength,
                x_potential=generated_data_discriminator_list_outputs[0],
                xx_potential=generated_data_discriminator_list_outputs[1]
            )

        def discriminator_loss(generator_output, real_data, generated_data_discriminator_list_outputs,
                               real_data_discriminator_list_outputs):
            # Note that in the case of SGAN-P the discriminators have to optimize (maximize) the entropic 2-Wasserstein
            # distances between real and generated data and between generated data and generated data, so we use
            # negative of  their sum as a loss function
            loss_xy = -losses.entropic_2Wasserstein_loss_semidual_from_potential(
                x=generator_output,
                y=real_data,
                epsilon=entropic_regularization_strength,
                x_potential=generated_data_discriminator_list_outputs[0]
            )
            loss_xx = -losses.entropic_2Wasserstein_loss_semidual_from_potential(
                x=generator_output,
                y=generator_output,
                epsilon=entropic_regularization_strength,
                x_potential=generated_data_discriminator_list_outputs[1]
            )
            return loss_xy + loss_xx

    elif GAN_loss_and_structure == 'EW2GAN':
        real_data_discriminator = Discriminator(n_neurons=GAN_n_hidden_layer_neurons,
                                                             data_dim=data_dimension)
        generated_data_discriminator = Discriminator(n_neurons=GAN_n_hidden_layer_neurons,
                                                                  data_dim=data_dimension)
        gan = GANwithMultipleDiscriminators(generator=generator,
                                                         real_data_discriminator_list=[real_data_discriminator],
                                                         generated_data_discriminator_list=[
                                                             generated_data_discriminator
                                                         ],
                                                         use_cuda=use_cuda)

        def generator_loss(generator_output, real_data, generated_data_discriminator_list_outputs,
                           real_data_discriminator_list_outputs):

            return losses.entropic_2Wasserstein_loss_primal_from_potentials(
                x=generator_output,
                y=real_data,
                epsilon=entropic_regularization_strength,
                x_potential=generated_data_discriminator_list_outputs[0],
                y_potential=real_data_discriminator_list_outputs[0],
                normalize_probability=False,
                normalize_distances=False
            )

        def discriminator_loss(generator_output, real_data, generated_data_discriminator_list_outputs,
                               real_data_discriminator_list_outputs):

            return -losses.entropic_2Wasserstein_loss_dual_from_potentials(
                x=generator_output,
                y=real_data,
                epsilon=entropic_regularization_strength,
                x_potential=generated_data_discriminator_list_outputs[0],
                y_potential=real_data_discriminator_list_outputs[0]
            )
    # EW2GAN-RealSD means that the discriminator is associated with the real data
    elif GAN_loss_and_structure == 'EW2GAN-RealSD':
        real_data_discriminator = Discriminator(n_neurons=GAN_n_hidden_layer_neurons, data_dim=data_dimension)
        gan = GANwithMultipleDiscriminators(generator=generator, real_data_discriminator_list=[real_data_discriminator],
                                            generated_data_discriminator_list=[],
                                            use_cuda=use_cuda)

        def generator_loss(generator_output, real_data, generated_data_discriminator_list_outputs,
                           real_data_discriminator_list_outputs):

            return losses.entropic_2Wasserstein_loss_semidual_from_potential(
                y=generator_output,
                x=real_data,
                epsilon=entropic_regularization_strength,
                x_potential=real_data_discriminator_list_outputs[0]
            )

        def discriminator_loss(generator_output, real_data, generated_data_discriminator_list_outputs,
                               real_data_discriminator_list_outputs):

            return -losses.entropic_2Wasserstein_loss_semidual_from_potential(
                y=generator_output,
                x=real_data,
                epsilon=entropic_regularization_strength,
                x_potential=real_data_discriminator_list_outputs[0]
            )
    # EW2GAN-FakeSD means that the discriminator is associated with the generated data
    elif GAN_loss_and_structure == 'EW2GAN-FakeSD':
        generated_data_discriminator = Discriminator(n_neurons=GAN_n_hidden_layer_neurons, data_dim=data_dimension)
        gan = GANwithMultipleDiscriminators(generator=generator, real_data_discriminator_list=[],
                                            generated_data_discriminator_list=[generated_data_discriminator],
                                            use_cuda=use_cuda)

        def generator_loss(generator_output, real_data, generated_data_discriminator_list_outputs,
                           real_data_discriminator_list_outputs):

            return losses.entropic_2Wasserstein_loss_semidual_from_potential(
                x=generator_output,
                y=real_data,
                epsilon=entropic_regularization_strength,
                x_potential=generated_data_discriminator_list_outputs[0]
            )

        def discriminator_loss(generator_output, real_data, generated_data_discriminator_list_outputs,
                               real_data_discriminator_list_outputs):

            return -losses.entropic_2Wasserstein_loss_semidual_from_potential(
                x=generator_output,
                y=real_data,
                epsilon=entropic_regularization_strength,
                x_potential=generated_data_discriminator_list_outputs[0]
            )

    elif GAN_loss_and_structure == 'WGAN':
        discriminator = Discriminator(n_neurons=GAN_n_hidden_layer_neurons,
                                                   data_dim=data_dimension)
        gan = GANwithMultipleDiscriminators(generator=generator,
                                                         real_data_discriminator_list=[discriminator],
                                                         generated_data_discriminator_list=[discriminator],
                                                         use_cuda=use_cuda)

        def generator_loss(generator_output, real_data, generated_data_discriminator_list_outputs,
                           real_data_discriminator_list_outputs):

            return losses.wasserstein_loss_from_potentials(
                x_potential=generated_data_discriminator_list_outputs[0],
                y_potential=real_data_discriminator_list_outputs[0]
            )

        def discriminator_loss(generator_output, real_data, generated_data_discriminator_list_outputs,
                               real_data_discriminator_list_outputs):
            if gradient_penalty_strength > 0:
                gradient_penalty = gradient_penalty_strength * losses.compute_gradient_penalty_for_wgan(
                    gan.discriminator_list[0],
                    real_data=real_data,
                    generated_data=generator_output)
            else:
                gradient_penalty = 0
            return -losses.wasserstein_loss_from_potentials(
                x_potential=generated_data_discriminator_list_outputs[0],
                y_potential=real_data_discriminator_list_outputs[0]
            ) + gradient_penalty

    else:
        raise ValueError("Unkhown GAN_loss_and_structure parameter")

    return gan, generator_loss, discriminator_loss