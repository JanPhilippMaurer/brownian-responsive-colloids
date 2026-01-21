# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 2024

@author: Jan-Philipp Maurer
"""

import numpy as np
from numba import njit, prange

#Includes different force functions from different fitted potentials
import force_functions




###########################################################
#                   Used FUnctions                       #
###########################################################
"""
    Different potential functions, def in force_function.py, used in the force Calculations.
"""

#Function s from LDL Fit and guess1
####################
property_force_Psi = force_functions.force_psi_gaussian_d_sigma
Translation_force = force_functions.force_phi_shell_core_hertzian_dr
property_force_phi = force_functions.force_phi_shell_core_hertzian_d_sigma



###########################################################
#                   Calculate Forces                      #
###########################################################


@njit(parallel=True)
def calculate_all_forces(
    x: np.ndarray,
    sigma: np.ndarray,
    particle_num: int,
    box_leng: float,
    neighbors: np.ndarray
):
    """
    Calculates all particle forces from their positions and particle sizes.

    Args:
        x (numpy.ndarray): Array of 3d position vectors
        sigma (numpy.ndarray): Array of particle diameters
        particle_num (int): total number of particles
        box_leng (float): Edge length of the simulation box
        neighbors (numpy.ndarray): Array of neighboring particles

    Returns:
        f_size (numpy.ndarray): Array of calculated property force vectors
        f (numpy.ndarray): Array of calculated translation force vectors
        virial_corr (float): Virial correlation term for later pressure calculation
    """

    f_size = np.zeros(particle_num)
    f = np.zeros((particle_num,3))
    virial_corr = 0

    for i in prange(particle_num):
        f_size[i] += property_force_Psi(sigma[i])

    for i in prange(particle_num):

        x_i = x[i]
        s_i = sigma[i]

        for j in neighbors[i]:
            if j == -1:
                break
            x_j = x[j]
            s_j = sigma[j]

            #calculating distance between particles
            rx = x_i[0] -x_j[0]
            ry = x_i[1] -x_j[1]
            rz = x_i[2] -x_j[2]
            #use periodic boundary conditions (PBC)
            #to find shortest distance between two particles
            rx -=  box_leng * np.round(rx / box_leng)
            ry -=  box_leng* np.round(ry / box_leng)
            rz -=  box_leng * np.round(rz / box_leng)
            r = np.sqrt(rx**2 + ry**2 +rz**2)
 
            #print(rx,ry,rz)
         
            #calculating translation force for i-th particle
            vec = np.empty(3)
            vec[0], vec[1], vec[2] = rx, ry, rz
            f_v = Translation_force(vec, r, s_i, s_j)
            
            if i < j:
                virial_corr += np.dot(vec, f_v)
                
            for d in range(3):
                    f[i,d] += f_v[d]
                    f[j,d] -= f_v[d]
            
            fs = property_force_phi(r, s_i, s_j)
            f_size[i] += fs
            f_size[j] += fs
    return f_size,f, virial_corr



