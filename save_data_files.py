# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13  2024

@author: Jan-Philipp Maurer
"""
import numpy as np
from datetime import date
from io import StringIO



def write_info_sheet(run_name: str,
                     delta_t: float,
                     N_equ: int,
                     N_dt: int,
                     save_sequence: int):
    """
        Creates a .txt file containing base parameters of the simulation as well as the name of the created simulation data files.
    Args:
        run_name (str): Name of the run
        delta_t (float): Time step size
        N_equ (int): Number of equilibration steps
        N_dt (int): Number of simulation time steps
        save_sequence:

    Returns:
        -
    """
    today = date.today().strftime(r"-%b-%d-%Y")

    # simulation settings
    # "sim-data/"+
    with open(run_name + "simulation-info.txt", "w") as file:
        file.write("###########################################\n")
        file.write("#           -" + run_name + "           #\n")
        file.write("###########################################\n")
        file.write("\n")
        file.write("This file contains Information of the simulation \""
                   + run_name + "\".\n")
        file.write("\n")
        file.write("Simulation Parameter\n")
        file.write("###################################\n")
        file.write("\n")
        file.write("Time step: " + f"{delta_t:.1e}\n")
        file.write("\n")
        file.write("Number of equilibration steps: " + f"{N_equ:.1e}\n")
        file.write("\n")
        file.write("Number of simulation steps: " + f"{N_dt:.1e}\n")
        file.write("\n")
        file.write("Save particle positions and size every " + str(save_sequence) + " steps.\n")
        file.write("\n")
        file.write("\n")
        file.write("Simulation data files for different packing fractions:\n")
        file.write("######################################################\n")
        file.close()




def save_timestep_ovito(file: str,
                        time: float,
                        particle_num: int,
                        box_bounds: np.ndarray,
                        positions: np.ndarray,
                        sigma: np.ndarray, ):
    """
    Function writing a timestep in Ovito readable style into a text file. For more context look into README.md.
    Args:
        file (str): name of the file
        time (float): simulation time
        particle_num (int): number of particles
        box_bounds (float): simulation box bounds in style [x_high, y_high, z_high]:
        positions (np.ndarray): Array of particle positions in form [[x1, y1, z1], [x2, y2, z2]], ...
        sigma (np.ndarray): Array of particle diameters

    Returns:
        -
    """

    buffer = StringIO()

    buffer.write("ITEM: TIMESTEP\n")
    buffer.write(f"{time}\n")

    buffer.write("ITEM: NUMBER OF ATOMS\n")
    buffer.write(f"{particle_num}\n")

    buffer.write("ITEM: BOX BOUNDS\n")
    buffer.write(str(0) + f" {box_bounds[0]} xlo xhi\n")
    buffer.write(str(0) + f" {box_bounds[1]} ylo yhi\n")
    buffer.write(str(0) + f" {box_bounds[2]} zlo zhi\n")

    buffer.write("ITEM: ATOMS id type x y z diameter\n")
    for i in range(particle_num):
        x, y, z = positions[i]
        d = sigma[i]
        buffer.write(f"{i}  1 {x} {y} {z} {d} \n")

    with open(file, "a") as f:
        f.write(buffer.getvalue())


def save_pressure(P, eta, file):
    buffer = StringIO()
    buffer.write(str(P) + " " + str(eta) + "\n")

    with open(file, "a") as f:
        f.write(buffer.getvalue())

