from losses import squared_pairwise_distances, entropic_2Wasserstein_loss_primal_from_potentials, \
    entropic_2Wasserstein_loss_dual_from_potentials, entropic_2Wasserstein_loss, sinkhorn_loss_primal_from_potentials, \
    sinkhorn_loss_dual_from_potentials, sinkhorn_loss, get_optimal_second_potential_for_entropic_2Wasserstein_loss,\
    entropic_2Wasserstein_loss_semidual_from_potential
import numpy as np
import torch

# Specify the inputs -- must be nxd tensors
# If n > 2 then the numerical approximation won't work
dim = 32
n_pts = 150
x = torch.tensor(np.random.randn(n_pts, dim))
y = torch.tensor(np.random.randn(n_pts, dim))
# Specify regularization
epsilon = 10
# Specify the number of points to iterate over (influences the accuracy of the approximation)
n_points = 1000


# The sinkhorn distance is found by iterating over a grid of possible probability matrices (this is fast since for
# 2 points  there is only 1 degree of freedom -- 2*probability matrix is doubly stochastic)
def entropic_wasserstein_distance_numeric(x, y, epsilon):
    if x.shape[0] != 2 or y.shape[0] != 2:
        print("erorr, tensors must be 2-dimensional")
        return 0
    d = squared_pairwise_distances(x, y)
    min_dist = np.inf
    # iterate over the probability matrices
    for p in np.linspace(0, 0.5, n_points):
        P = [[0.5 - p, p], [p, 0.5 - p]]
        P = torch.tensor(P)
        s = torch.sum(P * d) + epsilon * (2 * np.log(2) - torch.sum(-P * torch.log(P + (P == 0))))
        if s < min_dist:
            min_dist = s
    return min_dist


def sinkhorn_distance_numeric(x, y, epsilon):
    if x.shape[0] != 2 or y.shape[0] != 2:
        print("erorr, tensors must be 2-dimensional")
        return 0

    return entropic_wasserstein_distance_numeric(x, y, epsilon) - 0.5 * (
            entropic_wasserstein_distance_numeric(x, x, epsilon) + entropic_wasserstein_distance_numeric(y, y, epsilon))

print("\nCHECKING ENTROPIC DISTANCE CALCULATION")

EntropicW_distance_calculated = entropic_2Wasserstein_loss(x, y, epsilon)
EntropicW_distance_potentials = entropic_2Wasserstein_loss(x, y, epsilon, return_potentials=True)
if n_pts == 2:
    EntropicW_distance_expected = entropic_wasserstein_distance_numeric(x, y, epsilon)
    difference_entropic = np.abs(EntropicW_distance_expected - EntropicW_distance_calculated) / EntropicW_distance_expected
    print("The relative difference between the expected and calculated entropic 2-Wasserstein distances is {}".format(
        difference_entropic))
else:
    EntropicW_distance_expected = EntropicW_distance_calculated
    print("Number of samples is not 2, so the expected distance is the one calculated with geomloss")

# Compare distances calculated from potentials
potential_x, potential_y = EntropicW_distance_potentials
EntropicW_distance_dual_calculated_from_potentials = entropic_2Wasserstein_loss_dual_from_potentials(x, y, epsilon,
                                                                                                     potential_x,
                                                                                                     potential_y)
difference_entropic_dual_from_potentials = np.abs(EntropicW_distance_expected -
                                             EntropicW_distance_dual_calculated_from_potentials) / EntropicW_distance_expected
print("The relative difference between the expected entropic 2-Wasserstein distance and the one calculated from "
      "potentials (in dual form): {}".format(difference_entropic_dual_from_potentials))

EntropicW_distance_primal_calculated_from_potentials = entropic_2Wasserstein_loss_primal_from_potentials(x, y, epsilon,
                                                                                                         potential_x,
                                                                                                         potential_y,
                                                                                                         normalize_probability=True,
                                                                                                         normalize_distances=True)
difference_entropic_primal_from_potentials = np.abs(EntropicW_distance_expected -
                                             EntropicW_distance_primal_calculated_from_potentials) / EntropicW_distance_expected
print("The relative difference between the expected entropic 2-Wasserstein distance and the one calculated from "
      "potentials (in primal form): {}".format(difference_entropic_primal_from_potentials))

EntropicW_distance_semidual_calculated_from_x_potential = entropic_2Wasserstein_loss_semidual_from_potential(x, y,
                                                                                                             epsilon,
                                                                                                             potential_x)
difference_entropic_semidual = np.abs(EntropicW_distance_expected -
                                             EntropicW_distance_semidual_calculated_from_x_potential) / EntropicW_distance_expected
print("The relative difference between the expected entropic 2-Wasserstein distance and the one calculated from "
      "x potential (in semidual form): {}".format(difference_entropic_semidual))


print("CHECKING SINKHORN DISTANCE CALCULATION")
sinkhorn_distance_calculated = sinkhorn_loss(x, y, epsilon)
sinkhorn_distance_potentials = sinkhorn_loss(x, y, epsilon, return_potentials=True)

if n_pts == 2:
    sinkhorn_distance_expected = sinkhorn_distance_numeric(x, y, epsilon)
    difference_sinkhorn = np.abs(sinkhorn_distance_expected - sinkhorn_distance_calculated) / sinkhorn_distance_expected
    print("The relative difference between the expected and calculated Sinkhorn distances is: {}".format(difference_sinkhorn))
else:
    sinkhorn_distance_expected = sinkhorn_distance_calculated
    print("Number of samples is not 2, so the expected distance is the one calculated with geomloss")

sinkhorn_distance_dual_from_potentials = sinkhorn_loss_dual_from_potentials(x, y, epsilon, *sinkhorn_distance_potentials)
difference_sinkhorn_dual = np.abs(sinkhorn_distance_expected - sinkhorn_distance_dual_from_potentials) / sinkhorn_distance_expected
print("The relative difference between the expected Sinkhorn distance and the one calculated from potentials (in dual form): "
      "{}".format(difference_sinkhorn_dual))


sinkhorn_distance_primal_from_potentials = sinkhorn_loss_primal_from_potentials(x, y, epsilon, *sinkhorn_distance_potentials,
                                                                                normalize_probability=True)
difference_sinkhorn_dual = np.abs(sinkhorn_distance_expected - sinkhorn_distance_primal_from_potentials) / sinkhorn_distance_expected
print("The relative difference between the expected Sinkhorn distance and the one calculated from potentials (in primal form): "
      "{}".format(difference_sinkhorn_dual))
sinkhorn_y_potential_from_x_potential = get_optimal_second_potential_for_entropic_2Wasserstein_loss(x, y, epsilon,
                                                                                                    sinkhorn_distance_potentials[0])

print("CHECKING OPTIMAL POTENTIAL CALCULATION")
difference_potentials = np.linalg.norm(sinkhorn_y_potential_from_x_potential - sinkhorn_distance_potentials[2]) / \
                        np.linalg.norm(sinkhorn_distance_potentials[2])
print("The relative difference between the optimal Sinkhorn potential and the one calculated from the other potential: "
      "{}".format(difference_potentials))
