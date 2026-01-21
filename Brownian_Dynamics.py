# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7  2024

@author: Jan-Philipp Maurer
"""

import numpy as np
from numba import njit, prange

s0= 44.49264 


@njit(parallel=True)
def brownian_dynamics(sigma: np.ndarray,
                    x: np.ndarray,
                    f: np.ndarray,
                    f_size: np.ndarray,
                    delta_time: float,
                    particle_num: int,
                    box_length: float,
                    noise_sigma: np.ndarray,
                    noise_r: np.ndarray,
                    sigma0: float):
    """
    Updates the position and particle size of all particles using brownian motion dynamics
    Args:
        sigma (numpy.ndarray): Array of particle sizes
        x (numpy.ndarray): Array of 3d position vectors
        f (numpy.ndarray): Array of interaction forces
        f_size (numpy.ndarray): Array of proparty forces
        delta_time (float): Time step
        particle_num (int): total number of particles
        box_length (float): Edge length of the simulation box
        noise_sigma (numpy.ndarray): Array of brownian motion noise for the particle sizes
        noise_r (numpy.ndarray): Array of brownian motion noise for the particle positions
        sigma0 (float): Initial mean particle size in most swollen state

    Returns:
        new_sigma (numpy.ndarray): Array of updated particle sizes
        x (numpy.ndarray): Array of updated 3d position vectors
    """
    #constants
    alpha = 1.
    T = 1.
    k_B = 1.
    D_T = 1. / sigma0
    D_sigma = alpha * D_T
    
    #noise strength for brownian motion
    std_s = np.sqrt(2.0 * D_sigma * delta_time)
    std_r = np.sqrt(2.0 * D_T * delta_time)

    new_sigma = np.empty(particle_num)
    
    for i in prange(particle_num):
        #calc new sigma with BM
        new_sigma_i = sigma[i] + f_size[i] * delta_time * D_sigma / (k_B*T) + noise_sigma[i]*std_s
        #check the update size and ignor to small or to large size updates
        if 0.05*s0 <= new_sigma_i <= 2.0*s0:
            new_sigma[i] = new_sigma_i
        else:
            new_sigma[i] = sigma[i]
        #calc new position vector
        for j in range(3):
            x_new = x[i,j] + f[i,j] * delta_time * D_T / (k_B*T) + noise_r[i][j]*std_r
            x[i,j] = x_new % box_length
            
    return new_sigma, x







