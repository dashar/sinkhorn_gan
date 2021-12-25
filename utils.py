import argparse
import torch
import architecture
import losses


def float_or_none(input_str):
    """
    Type converter for float or Nans (to work with argparse)
    :param input_str: input string
    :return: either input_str converted to a float or None if the input string is "None". If input_str is neither None,
    nor a float, an error will be thrown
    """
    if input_str in ["None", "none", "NONE"]:
        return None
    else:
        try:
            return float(input_str)
        except TypeError:
            raise TypeError("only 'None', 'none' or 'NONE' strings and float values can be converted to float_or_none "
                            "values")


def bool_converter(input_str):
    """
    Type converter for boolean values (to work with argparse)
    :param input_str: input string
    :return: input_str converted to a boolean if the input string is "true, false" (case incensitive).
    An error will be thrown  otherwise
    """
    if input_str.lower() == 'true':
        return True
    elif input_str.lower() == 'false':
        return False
    else:
        raise TypeError("only 'true' or 'false' (case insensitive) strings can be converted to boolean values")


def create_input_parameters_parser():
    """
    Create a data parser to work with command line arguments
    :return: argparse.ArgumentParser variable with data, architecture, loss function and training parameters
    """

    architectures = ['SGAN-NP', 'SGAN-P', 'EW2GAN', 'WGAN', 'EW2GAN-RealSD', 'EW2GAN-FakeSD', 'SGAN-RealSD',
                     'SGAN-FakeSD']
    initializations = ['he', 'glorot']
    cuda_available = torch.cuda.is_available()
    optimizers = ['SGD', 'Adam', "RMSprop"]
    parser = argparse.ArgumentParser(description='Process training parameters')

    # Data parameters
    parser.add_argument('--data_dimension',                     default=32,             type=int,                   help='the real data dimension (the dimension of the output of the Generator ')
    parser.add_argument('--sample_size',                        default=100000,         type=int,                   help='the size of the input sample')
    parser.add_argument('--validation_sample_size',             default=1000,           type=int,                   help='the size of the validation sample (pass 0 for no validation)')
    parser.add_argument('--use_random_covariance',              default=True,           type=bool_converter,        help='if true, the real data has a random covariance with Frobenius norm 1, otherwise the covariance matrix is a scaled identity matrix with Frobenius norm 1')

    # Architecture parameters
    parser.add_argument('-GAN_loss_and_structure',                                      choices=architectures,      help='the GAN structure (number of discriminators) and the loss function')
    parser.add_argument('--latent_dimension',                   default=4,              type=int,                   help='the latent space dimension (the dimension of the input to the generator ')
    parser.add_argument('--GAN_n_hidden_layer_neurons',         default=64,             type=int,                   help='the dimension of the hidden layers in the discriminators and the generator of a GAN')
    parser.add_argument('--use_linear_generator',               default=False,          type=bool_converter,        help='if True, the Generator is linear, otherwise it is a 3-hidden-layer Neural Network')

    # Loss function parameters
    parser.add_argument('--entropic_regularization_strength',   default=12,             type=float,                 help='entropic regularization parameter (strength) of the GAN, this is the constant in front of the Mutual Information term, has to be 0 for a WGAN')
    parser.add_argument('--gradient_penalty_strength',          default=0,              type=float,                 help='constant which the gradient penalty is multiplied by in the loss function')

    # Training parameters
    parser.add_argument('--discriminator_clip_value',           default=None,           type=float_or_none,         help='the value to clip the discriminator weights to, the weights are clipped to to (-discriminator_clip_value, discriminator_clip_value)')
    parser.add_argument('--critic_iterations',                  default=5,              type=int,                   help='the number of critic (discriminator) iterations, for non-parametric Sinkhorn GAN this is always set to 0 regardless of the passed value')
    parser.add_argument('--batch_size',                         default=200,            type=int,                   help='the size of data batches for mini-batch gradient descent')
    parser.add_argument('--epochs',                             default=500,            type=int,                   help='the number of training epochs')
    parser.add_argument('--weight_initializer',                 default='he',           choices=initializations,    help='GAN weights initializer')
    parser.add_argument('--use_cuda',                           default=cuda_available, type=bool_converter,        help='whether to use cuda, torch.cuda.is_available() by default')
    parser.add_argument('--path_to_results_parent_folder',      default='./train',                                  help='path to the parent folder of the training results, the results will then be written to a folder named by the current date and time')

    # Optimizer parameters
    parser.add_argument('--optimizer',                          default='SGD',           choices=optimizers,       help='the type of oprimizer to use')
    parser.add_argument('--optimizer_learning_rate',            default=1e-4,            type=float,               help='the ooptimizer learning rate')
    parser.add_argument('--optimizer_beta1',                    default=0.5,             type=float,               help='if optimizer is Adam, the beta1 parameter, otherwise it is unused')
    parser.add_argument('--optimizer_beta2',                    default=0.9,             type=float,               help='if optimizer is Adam, the beta2 parameter, otherwise it is unused')
    parser.add_argument('--momentum',                           default=0.0,             type=float,               help='optimizer momentum (not used for Adam)')
    parser.add_argument('--alpha',                              default=0.99,            type=float,               help='if optimizer is RMSprop, the alpha (smoothing) parameter')

    return parser


def weights_initializer_fn(weight_initializer_str):
    """
    Helper function to create a neural network weight initializer function from the initializer name. The biases are
    filled with 0s, He and Glorot initializers are supported for now. The function is intended to be used with
    torch.nn.Module.apply as it initializes all weights and biases of torch.nn.Linear layers
    :param weight_initializer_str: the name of the initialization to be used, 'He' and 'Glorot' (case insensitive) are
    supported for now
    :return: weight initializer function that applies torch.nn.init.kaiming_uniform_ or torch.nn.init.xavier_uniform_
    to the weight attribute of the input if the input is an instance of torch.nn.Linear
    """
    if weight_initializer_str.lower() == 'he':
        def weight_initializer(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                m.bias.data.fill_(0.0)
    elif weight_initializer_str.lower() == 'glorot':
        def weight_initializer(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('relu'))
                m.bias.data.fill_(0.0)
    else:
        raise ValueError("Unkhown weight_initializer parameter, only 'he' and 'glorot' are supported")
    return weight_initializer


