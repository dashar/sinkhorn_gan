import os, sys
sys.path.append(os.getcwd())

import random
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sklearn.datasets


import tflib as lib
import tflib.ops.linear
import tflib.plot



#PARAMETERS
MODE = 'wgan-gp' # wgan or wgan-gp
DATASET = 'gaussian' # 8gaussians, 25gaussians, swissroll
DIM = 64 # Model dimensionality (number of neurons in the hidden layer(s))
FIXED_GENERATOR = False # whether to hold the generator fixed at real data plus
                        # Gaussian noise, as in the plots in the paper
LAMBDA = .1 # Smaller lambda makes things faster for toy tasks, but isn't
            # necessary if you increase CRITIC_ITERS enough
CRITIC_ITERS = 5 # How many critic iterations per generator iteration
BATCH_SIZE = 256 # Batch size
ITERS = 20000#100000 # how many generator iterations to train for
DATA_DIM = 32
LATENT_DIM = 4
INITIALIZATION = 'glorot'#'he'#
COVARIANCE_SCALE = np.sqrt(DATA_DIM)
INITIALIZE_LAST = True
SAMPLE_SIZE = 100000

lib.print_model_settings(locals().copy())

#whether to initialize the last layer (linear layer) as INITIALIZATION or a random orthogonal matrix
init_last = INITIALIZATION if INITIALIZE_LAST else None

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear',
        n_in,
        n_out,
        inputs,
        initialization=INITIALIZATION
    )
    output = tf.nn.relu(output)
    return output

def Generator(n_samples, real_data):
    if FIXED_GENERATOR:
        return real_data + (1.*tf.random_normal(tf.shape(real_data)))
    else:
        noise = tf.random_normal([n_samples, LATENT_DIM])
        output = ReLULayer('Generator1', LATENT_DIM, DIM, noise)
        output = ReLULayer('Generator2', DIM, DIM, output)
        output = ReLULayer('Generator3', DIM, DIM, output)
        output = lib.ops.linear.Linear('Generator4', DIM, DATA_DIM, output, initialization=init_last)#MAYBE THEY DIDN'T DO IT
        return output

def Discriminator(inputs):
    output = ReLULayer('Discriminator1', 32, DIM, inputs)
    output = ReLULayer('Discriminator2', DIM, DIM, output)
    output = ReLULayer('Discriminator3', DIM, DIM, output)
    output = lib.ops.linear.Linear('Discriminator4', DIM, 1, output, initialization=init_last)
    return tf.reshape(output, [-1])

real_data = tf.placeholder(tf.float32, shape=[None, DATA_DIM])
fake_data = Generator(BATCH_SIZE, real_data)
disc_real = Discriminator(real_data)
disc_fake = Discriminator(fake_data)

# WGAN loss
disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
gen_cost = -tf.reduce_mean(disc_fake)

# WGAN gradient penalty
if MODE == 'wgan-gp':
    alpha = tf.random_uniform(
        shape=[BATCH_SIZE,1], 
        minval=0.,
        maxval=1.
    )
    interpolates = alpha*real_data + ((1-alpha)*fake_data)
    disc_interpolates = Discriminator(interpolates)
    gradients = tf.gradients(disc_interpolates, [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1)**2)
 
    disc_cost += LAMBDA*gradient_penalty

disc_params = lib.params_with_name('Discriminator')
gen_params = lib.params_with_name('Generator')

if MODE == 'wgan-gp':
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4, 
        beta1=0.5, 
        beta2=0.9
    ).minimize(
        disc_cost, 
        var_list=disc_params
    )
    if len(gen_params) > 0:
        gen_train_op = tf.train.AdamOptimizer(
            learning_rate=1e-4, 
            beta1=0.5, 
            beta2=0.9
        ).minimize(
            gen_cost, 
            var_list=gen_params
        )
    else:
        gen_train_op = tf.no_op()

else:
    disc_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(
        disc_cost, 
        var_list=disc_params
    )
    if len(gen_params) > 0:
        gen_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(
            gen_cost, 
            var_list=gen_params
        )
    else:
        gen_train_op = tf.no_op()


    # Build an op to do the weight clipping
    clip_ops = []
    for var in disc_params:
        clip_bounds = [-.01, .01]
        clip_ops.append(
            tf.assign(
                var, 
                tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
            )
        )
    clip_disc_weights = tf.group(*clip_ops)
    
print("Generator params:")
for var in lib.params_with_name('Generator'):
    print("\t{}\t{}".format(var.name, var.get_shape()))
print("Discriminator params:")
for var in lib.params_with_name('Discriminator'):
    print("\t{}\t{}".format(var.name, var.get_shape()))

frame_index = [0]

#DATASET GENERATOR
def inf_train_gen():
    if DATASET == 'gaussian':
        np.random.seed(10)
        full_dataset = np.random.randn(SAMPLE_SIZE,DATA_DIM) / np.sqrt(COVARIANCE_SCALE) 
        i = 0
        offset = 0
        while True:
            dataset = full_dataset[i*BATCH_SIZE+offset:(i+1)*BATCH_SIZE+offset,:]
            if (i+1)*BATCH_SIZE+offset > SAMPLE_SIZE: 
                offset = (i+1)*BATCH_SIZE+offset - SAMPLE_SIZE
                np.random.shuffle(full_dataset)
                dataset = np.concatenate([dataset,full_dataset[:offset,:]], axis = 0)
                i = -1 
            i+=1
            yield dataset
            
mean_fake_data = tf.reduce_mean(fake_data, axis=0, keep_dims=True)
vx = tf.matmul(tf.transpose(fake_data),fake_data)/tf.cast(tf.shape(fake_data)[0]-1, tf.float32)
mx = tf.matmul(tf.transpose(mean_fake_data), mean_fake_data)
fake_data_covariance = vx - mx

ML_covariance = np.eye(DATA_DIM)/COVARIANCE_SCALE
accuracy_metric = tf.norm(fake_data_covariance - ML_covariance)

# Train loop!
accuracy_history = []

#I chose covariance_scale = np.sqrt(DATA_DIM)
#if COVARIANCE_SCALE == DATA_DIM:
#    model_name = "d_"
#else:
#    model_name = "root_d_"
#I chose to initialize the last layer
#model_name = model_name + "initialize_last_" + str(INITIALIZE_LAST) + "_initialization_" + INITIALIZATION
if MODE == 'wgan-gp':
    model_name = "WGAN_GP"
else:
    model_name = "WGAN_WC"
model_name = model_name + "_LATENT_DIM_" + str(LATENT_DIM) + "_initialization_" + INITIALIZATION


plt.figure()
with tf.Session() as session:
    session.run(tf.initialize_all_variables())
    gen = inf_train_gen()
    for iteration in range(ITERS):
        # Train generator
        if iteration > 0:
            _ = session.run(gen_train_op)
        # Train critic
        for i in range(CRITIC_ITERS):
            _data = next(gen)
            _disc_cost, _, accuracy = session.run(
                [disc_cost, disc_train_op, accuracy_metric],
                feed_dict={real_data: _data}
            )
            if MODE == 'wgan':
                _ = session.run([clip_disc_weights])
        # Write logs and save samples
        #print(np.abs(get_cov_diff(fake_sample)-accuracy)/get_cov_diff(fake_sample))
        lib.plot.plot('disc cost', _disc_cost)
        lib.plot.plot('accuracy', accuracy)
        accuracy_history.append(accuracy)
        if iteration % 100 == 99:
            lib.plot.flush("./"+model_name+".pkl")
            plt.clf()
            plt.grid("on", "both")
            plt.plot(np.arange(iteration+1), accuracy_history)
            plt.plot(np.arange(iteration+1), np.zeros(iteration+1))
            plt.savefig("./accuracy_history_"+model_name+".png")
        lib.plot.tick()


lib.plot.flush("./"+model_name+".pkl")

step = 50
plt.clf()
plt.grid("on", "both")
accuracy_history = np.array(accuracy_history)
if np.max(accuracy_history) <= 3.5:
    idx = 0
else:
    idx = np.max(np.where(accuracy_history>3.5))+1
plt.plot(np.arange(idx,iteration+1,step), accuracy_history[idx::step], linewidth=2, color = 'red')
plt.plot(np.arange(iteration+1), np.zeros(iteration+1), linewidth=2, color = 'green')
plt.savefig("./accuracy_history_"+model_name+".png")