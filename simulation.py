
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 2024

@author: Jan-Philipp Maurer
"""
import numpy as np


#imported functions
from simulation_functions import equilibration, production_Run,particle_grid

from save_data_files import write_info_sheet, save_pressure

# Get today's date








def main(packing_fraction: list,
         file_names: str,
         sigma0: float,
         delta_time: float,
         N_equ: int,
         N_dt: int,
         save_sequence: int):
    """

    Args:
        packing_fraction (list): List of packing fractions
        file_names (str): Main name of the simulation files
        sigma0 (float): Initial mean particle diameter in most swollen state
        delta_time: Time step size
        N_equ (int): Number of equilibration time steps
        N_dt (int): Number of time steps of the final simulation run
        save_sequence (int): number of time steps after which the particle position etc., are saved

    Returns:
        -
    """



    write_info_sheet(file_names, delta_time, N_equ, N_dt, save_sequence)

    grid_width, grid_depth, grid_height = 4,5,5
    particle_num = grid_width * grid_height * grid_depth
    box_length = ((4/3) * particle_num * np.pi *  (0.5*sigma0)**3 / packing_fraction[0])**(1/3)
    x = particle_grid(box_length,grid_width, grid_depth,grid_height)
    sigma = np.ones(particle_num) * sigma0 #starting sizes
    
    #equilibrate the system from grid starting points
    sigma, x = equilibration(N_equ, sigma, x, particle_num, delta_time, box_length, sigma0)
    

    for eta in packing_fraction:
        #hydrodynamic radius
        r_H = 0.5*sigma0
        #new simulation box dimensions for different packing fraction
        box_length = ((4/3) * particle_num * np.pi *  (r_H)**3 / eta)**(1/3)
        
        eta1 = (4/3) * particle_num * np.pi * r_H**3 / box_length**3
        print("#######################################")
        print("packing fraction: ", eta1)
        print("#######################################")
        sim_file_name =  file_names + str(eta) + r".txt"
        #writes down the name of the current run, made of date and packing fraction
        with open(file_names+ "simulation-info.txt", "a") as file:
            file.write(sim_file_name + "\n")

        # for different packing fraction starting point is particle positions of packing fraction before to save another
        # equilibration of the system
        x, sigma, P = production_Run(N_dt, sigma,x, particle_num,delta_time,
                                     box_length,save_sequence,sim_file_name,sigma0)
        #saves the calculated system Pressure together with the responding packing fraction in a .txt file
        save_pressure(P, eta, file_names+"pressure.txt")



