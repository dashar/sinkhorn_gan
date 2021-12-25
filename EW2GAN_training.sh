#!/bin/sh

FILE=./bin/activate
if test -f "$FILE"; then
    source ./bin/activate
fi

# An example of a script for training EW2GAN
python main.py                                    \
    `# Data parameters`                           \
    --data_dimension 32                           \
    --sample_size 100000                          \
    --validation_sample_size 10000                \
    --use_random_covariance True                  \
    `# Architecture parameters`                   \
    -GAN_loss_and_structure EW2GAN                \
    --latent_dimension 4                          \
    --GAN_n_hidden_layer_neurons 64               \
    --use_linear_generator False                  \
    `# Loss function parameters`                  \
    --entropic_regularization_strength 12         \
    --gradient_penalty_strength 0                 \
    `# Training parameters`                       \
    --discriminator_clip_value None               \
    --critic_iterations 5                         \
    --batch_size 200                              \
    --epochs 500                                  \
    --weight_initializer he                       \
    `# --use_cuda False  # uncomment if want to specify cuda usage, used if available by default` \
    --path_to_results_parent_folder ./train       \
    `# Optimizer parameters`                      \
    --optimizer SGD                               \
    --optimizer_learning_rate 1e-3                \
    --optimizer_beta1 0.1                         \
    --optimizer_beta2 0.18                        \
    --momentum 0.9                                \
    --alpha 0.99
