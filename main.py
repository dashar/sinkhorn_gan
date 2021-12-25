import os
import sys

import datetime
import numpy as np
import torch

from torch.utils import tensorboard
import architecture
import utils
import data_processing
import metrics

import pickle as pkl

# import time processing libraries to get the runtime
import time
from datetime import timedelta

sys.path.append(os.getcwd())

#
# def create_all_covariance_metrics(dataset, covariance_calculator, args, gan):
#     def batch_covariance_distance_to_optimal_cov(generator_output, real_data,
#                                                  generated_data_discriminator_list_outputs,
#                                                  real_data_discriminator_list_outputs):
#         return metrics.frobenius_distance_to_generator_output_covariance(
#             metrics.get_optimal_output_covariance_gaussian_sgan(dataset.cov, args.latent_dim),
#             generator_output,
#             covariance_calculator,
#             gan,
#             permute_target_matrix_diagonal=not args.use_random_covariance,
#             use_sample_covariance=True
#         )
#
#     def batch_covariance_distance_to_true_cov(generator_output, real_data,
#                                               generated_data_discriminator_list_outputs,
#                                               real_data_discriminator_list_outputs):
#         return metrics.frobenius_distance_to_generator_output_covariance(
#             dataset.cov,
#             generator_output,
#             covariance_calculator,
#             gan,
#             permute_target_matrix_diagonal=not args.use_random_covariance,
#             use_sample_covariance=True
#         )
#
#     def running_covariance_distance_to_optimal_cov(generator_output, real_data,
#                                                    generated_data_discriminator_list_outputs,
#                                                    real_data_discriminator_list_outputs):
#         return metrics.frobenius_distance_to_generator_output_covariance(
#             metrics.get_optimal_output_covariance_gaussian_sgan(dataset.cov, args.latent_dim),
#             generator_output,
#             covariance_calculator,
#             gan,
#             permute_target_matrix_diagonal=not args.use_random_covariance,
#             use_sample_covariance=False
#         )
#
#     def running_covariance_distance_to_true_cov(generator_output, real_data,
#                                                 generated_data_discriminator_list_outputs,
#                                                 real_data_discriminator_list_outputs):
#         return metrics.frobenius_distance_to_generator_output_covariance(
#             dataset.cov,
#             generator_output,
#             covariance_calculator,
#             gan,
#             permute_target_matrix_diagonal=not args.use_random_covariance,
#             use_sample_covariance=False
#         )
#
#     return batch_covariance_distance_to_true_cov, batch_covariance_distance_to_optimal_cov, \
#            running_covariance_distance_to_true_cov, running_covariance_distance_to_optimal_cov
#
#     def batch_covariance_difference(generator_output, real_data, generated_data_discriminator_list_outputs,
#                                     real_data_discriminator_list_outputs):
#         return torch.tensor(np.linalg.norm(covariance_calculator.update(generator_output, return_batch_covariance=True)
#                                            - dataset.cov))
#
#     def running_covariance_difference(generator_output, real_data, generated_data_discriminator_list_outputs,
#                                       real_data_discriminator_list_outputs):
#         return torch.tensor(np.linalg.norm(covariance_calculator.update(generator_output, return_batch_covariance=False)
#                                            - dataset.cov))
#
#     val_covariance_metric = metrics.CovarianceMatrixCalculator()
#
#     def val_running_covariance_difference(generator_output, real_data, generated_data_discriminator_list_outputs,
#                                           real_data_discriminator_list_outputs):
#         val_covariance_metric.set_updated_to_false()
#         return torch.tensor(np.linalg.norm(covariance_calculator.update(generator_output, return_batch_covariance=False)
#                                            - dataset.cov))

if __name__ == "__main__":
    # Start the timer
    start_time = time.monotonic()

    # Get the command line parameters
    parser = utils.create_input_parameters_parser()
    args = parser.parse_args()

    # Create a directory for output results named with the current date and time
    torch.manual_seed(1)
    np.random.seed(1)
    results_folder_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")

    if not os.path.exists(args.path_to_results_parent_folder):
        os.mkdir(args.path_to_results_parent_folder)

    results_path = os.path.join(args.path_to_results_parent_folder, results_folder_name)
    os.mkdir(results_path)

    # Save all the input parameters to the results_path directory
    with open(os.path.join(results_path, 'training_parameters.txt'), 'w') as f:
        for key, val in args.__dict__.items():
            f.write("{}: {}\n".format(key, val))

    # Create the training dataset and a data loader
    dataset = data_processing.GaussianDataset(sample_size=args.sample_size, data_dim=args.data_dimension,
                                              latent_dim=args.latent_dimension,
                                              use_random_covariance=args.use_random_covariance)
    real_data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Create the validation dataset and a data loader
    if args.validation_sample_size > 0:
        validation_dataset = data_processing.GaussianDataset(sample_size=args.validation_sample_size,
                                                             data_dim=args.data_dimension,
                                                             latent_dim=args.latent_dimension,
                                                             covariance_to_use=dataset.cov)
        validation_real_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size,
                                                                  shuffle=True)
    else:
        validation_real_data_loader = None

    # Create a latent data sampling function
    def latent_data_fn(batch_size):
        return torch.Tensor(np.random.randn(batch_size, args.latent_dimension)).float()

    # Create the GAN neural network and its losses
    # gan, generator_loss, discriminator_loss = utils.create_gan_from_params(**args)
    gan, generator_loss, discriminator_loss = architecture.create_gan_from_params(
        GAN_loss_and_structure=args.GAN_loss_and_structure,
        data_dimension=args.data_dimension,
        latent_dimension=args.latent_dimension,
        GAN_n_hidden_layer_neurons=args.GAN_n_hidden_layer_neurons,
        use_linear_generator=args.use_linear_generator,
        entropic_regularization_strength=args.entropic_regularization_strength,
        gradient_penalty_strength=args.gradient_penalty_strength,
        use_cuda=args.use_cuda
    )

    # Do weights initialization
    weight_initializer = utils.weights_initializer_fn(args.weight_initializer)

    gan.generator.apply(weight_initializer)
    for discriminator in gan.discriminator_list:
        discriminator.apply(weight_initializer)

    # Define optimizers
    discriminators_parameters = sum([list(discriminator.parameters()) for discriminator in gan.discriminator_list], [])
    if args.optimizer == "SGD":
        discriminator_optimizer = torch.optim.SGD(discriminators_parameters, lr=args.optimizer_learning_rate,
                                                  momentum=args.momentum) if len(discriminators_parameters)>0 else None
        generator_optimizer = torch.optim.SGD(gan.generator.parameters(), lr=args.optimizer_learning_rate,
                                              momentum=args.momentum)
    elif args.optimizer == "Adam":
        discriminator_optimizer = torch.optim.Adam(discriminators_parameters, lr=args.optimizer_learning_rate,
                                                   betas=(args.optimizer_beta1, args.optimizer_beta2))  \
            if len(discriminators_parameters)>0 else None
        generator_optimizer = torch.optim.Adam(gan.generator.parameters(), lr=args.optimizer_learning_rate,
                                               betas=(args.optimizer_beta1, args.optimizer_beta2))
    elif args.optimizer == "RMSprop":
        discriminator_optimizer = torch.optim.RMSprop(discriminators_parameters, lr=args.optimizer_learning_rate,
                                                      alpha=args.alpha) if len(discriminators_parameters)>0 else None
        generator_optimizer = torch.optim.RMSprop(gan.generator.parameters(), lr=args.optimizer_learning_rate,
                                                  alpha=args.alpha)
    else:
        raise ValueError("Unknown optimizer")

    # Define the covariance metric
    covariance_calculator = metrics.CovarianceMatrixCalculator()
    covariance_metrics_dict = metrics.create_all_covariance_metrics(dataset.cov, covariance_calculator, gan,
                                                                    args.latent_dimension,
                                                                    (not args.use_random_covariance),
                                                                    SGAN_model="SGAN" in args.GAN_loss_and_structure,
                                                                    epsilon=args.entropic_regularization_strength)

    # Define the covariance metric for the validation dataset
    val_covariance_calculator = metrics.CovarianceMatrixCalculator()
    val_covariance_metrics_dict = metrics.create_all_covariance_metrics(dataset.cov, val_covariance_calculator, gan,
                                                                        args.latent_dimension,
                                                                        (not args.use_random_covariance),
                                                                        SGAN_model="SGAN" in args.GAN_loss_and_structure,
                                                                        epsilon=args.entropic_regularization_strength)
    # delete the batch metrics for validation dataset
    del val_covariance_metrics_dict['batch covariance distance to true cov']
    del val_covariance_metrics_dict['batch covariance distance to optimal cov']

    # Compile the gan with the created losses and the covariance difference metrics. Note that we clip all the
    # discriminators if args.discriminator_clip_value is not None
    discriminators_to_clip_weights = None if args.discriminator_clip_value is None else gan.discriminator_list
    clip_value = (-args.discriminator_clip_value, args.discriminator_clip_value) \
        if args.discriminator_clip_value is not None else None

    gan.compile(discriminator_loss, generator_loss, discriminator_optimizer, generator_optimizer,
                discriminator_step_metrics=None,
                generator_step_metrics=covariance_metrics_dict,
                validation_metrics=val_covariance_metrics_dict,
                metric_names_not_to_average=['running covariance distance to true cov',
                                             'running covariance distance to optimal cov'],
                discriminators_to_clip_weights=discriminators_to_clip_weights, clip_value=clip_value)

    # Create the tensorboard summary writer to log the training progress
    writer = tensorboard.SummaryWriter(os.path.join(results_path, "tensorboard_logs"))
    out = gan.train(latent_data_fn, real_data_loader, args.epochs,
                    functions_to_run_at_epoch_start=[covariance_calculator.reset, val_covariance_calculator.reset],
                    functions_to_run_at_generator_step_start=[covariance_calculator.set_updated_to_false],
                    use_new_real_data_for_generator_step=args.GAN_loss_and_structure in ['SGAN-P', 'SGAN-NP', 'EW2GAN'],
                    use_new_latent_data_for_generator_step=args.GAN_loss_and_structure == 'SGAN-NP',
                    discriminator_iterations_per_generator_iterations=args.critic_iterations,
                    validation_real_data_loader=validation_real_data_loader,
                    tensorboard_writer=writer, n_generator_iterations_to_print_batch_metrics=None)

    writer.flush()
    writer.close()

    # end the timer and save the time elapsed
    end_time = time.monotonic()
    elapsed_time_str = timedelta(seconds=end_time - start_time).__str__()

    # create the results dictionary
    results_dict = {'iterationwise_generator_step_metrics': out[0],
                    'iterationwise_discriminator_step_metrics': out[1],
                    'epoch_averaged_discriminator_step_metrics': out[2],
                    'epoch_averaged_generator_step_metrics': out[3],
                    'epoch_validation_step_metrics': out[4],
                    'execution time': elapsed_time_str,
                    'data covariance': dataset.cov}

    # Dump the training results into a pkl file
    with open(os.path.join(results_path, 'training_results.pkl'), 'wb') as f:
        pkl.dump(results_dict, f)
