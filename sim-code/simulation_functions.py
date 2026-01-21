# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13  2024

@author: Jan-Philipp Maurer
"""

import numpy as np
from numba import njit, prange

from force_calculation import calculate_all_forces
from Brownian_Dynamics import brownian_dynamics
from save_data_files import save_timestep_ovito
#from main import sigma0
#max distance for neighbor list


def particle_grid(box_length: float,
                  grid_width: int,
                  grid_depth: int,
                  grid_height: int):
    """
        Creates an Array with the position vectors of all particles. They are thereby spred in a cuboid grid.
    Args:
        box_length (float): Edge length of the simulation box
        grid_width (int): Number of particles in the width
        grid_depth (int): Number of particles in the depth
        grid_height (int): Number of particles in the height

    Returns:
        x (np.array): Array of position vectors
    """
    lim = box_length - 1.
    x = np.linspace(0.1, lim, grid_width)
    y = np.linspace(0.1, lim, grid_depth)
    z = np.linspace(0.1, lim, grid_height)

    x_position, y_position, z_position = np.meshgrid(x, y, z)

    x_position = x_position.flatten()
    y_position = y_position.flatten()
    z_position = z_position.flatten()

    x = []
    for i in range(len(x_position)):
        v = np.array([x_position[i], y_position[i], z_position[i]])
        x.append(v)
    x = np.array(x)
    return x



@njit(parallel=True)
def neighbor_list(x: np.ndarray,
                  particle_num: int,
                  box_length: float,
                  sigma0: float):
    """
    Defines a list of neighbors in

    Args:
        x (numpy.ndarray): Array of 3d position vectors
        particle_num (int): total number of particles
        box_length (float): Edge length of the simulation box
        sigma0 (float): Initial mean particle size in most swollen state

    Returns:
    neighbors (numpy.ndarray): Array of 3d position vectors of neighboring particles
    """
    max_neighbors = 50
    #tag the empty neighbor places with -1 (indices always >0)
    neighbors = -1 * np.ones((particle_num, max_neighbors), dtype=np.int32)
    #because we have only interaction when overlapping, small cut
    cutoff = sigma0 + 0.1 * sigma0
    cutoff_sq = cutoff ** 2

    for i in prange(particle_num):
        xi = x[i]
        count = 0
        for j in range(particle_num):
            if i == j:
                continue
            xj = x[j]
            dx = xi[0] - xj[0]
            dy = xi[1] - xj[1]
            dz = xi[2] - xj[2]

            dx -= box_length * np.round(dx / box_length)
            dy -= box_length * np.round(dy / box_length)
            dz -= box_length * np.round(dz / box_length)

            dist_sq = dx*dx + dy*dy + dz*dz
            if dist_sq < cutoff_sq and count < max_neighbors:
                neighbors[i, count] = j
                count += 1
    return neighbors


#calculates max displacement
@njit
def max_dis(x, x_old, box_leng):
    displacements = x - x_old
    displacements -= box_leng * np.round(displacements / box_leng)
    squared_displacements = np.sum(displacements**2, axis=1)
    max_displacement = np.sqrt(np.max(squared_displacements))
    return max_displacement








def equilibration(N_equ: int
                  ,sigma: np.ndarray,
                  x: np.ndarray,
                  particle_num: int,
                  delta_time: float,
                  box_length: float,
                  sigma0: float):
    """
    Runs the simulation for a given number of equilibration steps.
    Args:
        N_equ (int): Number of equilibration time steps
        sigma (numpy.ndarray): Array of particle sizes
        x (numpy.ndarray): Array of position vectors
        particle_num (int): Number of particles
        delta_time (float): Time step size
        box_length (float): Edge length of the simulation box
        sigma0 (float): Initial mean particle size in most swollen state

    Returns:
        sigma (numpy.ndarray): Array of particle sizes after equilibration run
        x (numpy.ndarray): Array of position vectors after equilibration run
    """
    print('______Equilibration of the System______')
    
    neighbors= neighbor_list(x, particle_num, box_length,sigma0)
    x_old = np.copy(x)
    #for i in tqdm(range(N_equ)):
    for i in range(N_equ):
        #statistical nois for Brownian dynamics
        noise_sigma = np.random.normal(0, 1, size=particle_num)
        noise_r = np.random.normal(0, 1, size=(particle_num, 3))
        
        
        
       
        
        #calculate forces
        f_size,f, virial_corr = calculate_all_forces(x, sigma, particle_num, box_length,neighbors)
        #update position and size
        sigma, x = brownian_dynamics(sigma, x, f, f_size, delta_time, particle_num,box_length,noise_sigma,noise_r,sigma0)
        
        
        
        max_displacement = max_dis(x, x_old, box_length)
        buffer = 0.5 * sigma0
        if max_displacement > buffer:
            neighbors = neighbor_list(x, particle_num, box_length,sigma0)
            x_old = np.copy(x)
    return sigma, x




def production_Run(N_dt: int,
                   sigma: np.ndarray,
                   x: np.ndarray,
                   particle_num: int,
                   delta_time: float,
                   box_length: float,
                   save_sequence: int,
                   file: str,
                   sigma0: float):
    """
    Main simulation run done for e given number of timesteps N_dt.
    Args:
        N_dt (int): Number of time steps of the final simulation run
        sigma (numpy.ndarray): Array of particle diameters
        x (numpy.ndarray): Array of position vectors
        particle_num (int): Number of particles
        delta_time (float): Time step size
        box_length (float): Edge length of the simulation box
        save_sequence (int): number of time steps after which the particle position etc., are saved
        file (str): Main name of the saved files
        sigma0 (float): Initial mean particle diameter in most swollen state

    Returns:
        x (numpy.ndarray): Array of position vectors after the simulation run
        sigma (numpy.ndarray): Array of particle sizes after the simulation run
        Pressure (float): calculated system pressure.
    """
    print('______Production Run______')
    
    
    time = 0
    box_bounds = np.ones(3) * box_length
    virial_sum = 0
        
    save_timestep_ovito(file, time, particle_num, box_bounds, x, sigma)
    
    neighbors= neighbor_list(x, particle_num, box_length,sigma0)
    x_old = np.copy(x)
    
    #for i in tqdm(range(N_dt), desc="Production"):
    for i in range(N_dt):
        # statistical nois for Brownian dynamics
        noise_sigma = np.random.normal(0, 1, size=particle_num)
        noise_r = np.random.normal(0, 1, size=(particle_num, 3))
        #calculate forces
        f_size,f, virial_corr = calculate_all_forces(x, sigma, particle_num, box_length,neighbors)
        #update position and particle size with Brownian dynamics
        sigma, x = brownian_dynamics(sigma, x, f, f_size, delta_time, particle_num,box_length,noise_sigma, noise_r, sigma0)
        
        max_displacement = max_dis(x, x_old, box_length)
        buffer = 0.5 * sigma0
        if max_displacement > buffer:
            neighbors = neighbor_list(x, particle_num, box_length,sigma0)
            x_old = np.copy(x)

        
        virial_sum += virial_corr
        time += delta_time

        #save particle information after a given sequence save_sequence
        if (i+1) % save_sequence == 0:
            save_timestep_ovito(file, time, particle_num, box_bounds, x, sigma)
    V = box_length**3
    Pressure = (particle_num / V) + (1 / (3 * V)) * (virial_sum / N_dt)
    return x, sigma, Pressure



