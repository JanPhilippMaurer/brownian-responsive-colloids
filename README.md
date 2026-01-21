# Brownian Dynamics Simulation of Responsive Colloids

This repository contains a Python implementation of a Brownian dynamics
simulation for stimulus-responsive colloidal particles. The code was
developed as part of the Master’s thesis *“Coarse-Graining of Responsive
Colloids”* (Jan-Philipp Maurer, November 2025).


## Physical Model

The dynamics of each colloidal particle are described by Brownian motion
according to the following discrete equation of motion:

$$
    \mathbf{r}(t+\Delta t) = \mathbf{r}(t)
    + \frac{D_{\mathrm T}}{k_{\mathrm B}T}\,\mathbf{F}(t)\,\Delta t
    + \mathbf{\xi},
  $$

where $\mathbf{r}(t)$ denotes the three-dimensional particle position,
$\Delta t$ is the time step, and $D_{\mathrm T}$ is the translational diffusion
coefficient. The thermal energy is given by $k_{\mathrm B}T$,
$\mathbf{F}(t)$ denotes the interaction force, and
$\mathbf{\xi}$ represents the stochastic displacement due to Brownian motion. Thereby 
$\mathbf{\xi}$ is drawn from a normal distribution with mean $0$ and 
a variance of $2D_{\mathrm T}\Delta t$.

## Code Structure

    sim-responsive-colloids/
    ├─ simulation.py                # main(...) function
    ├─ simulation_functions.py      # simulation for each timestep
    ├─ brownian_dynamics.py         # time integration
    ├─ force_calculation.py         # calculating distances and all forces
    ├─ force_functions.py           # translation and property force functions
    ├─ save_data_files.py           # save the simulation data
    ├─ run_simulation.py            # run main(...) with test parameters
    ├─ requirements.txt             # required Python packages
    └─ README.md


## Requirements

- Python ≥ 3.12.7
- numpy
- numba
- datetime
- io
- time (only for run_simulation.py)        

Dependencies can be installed with:

    pip install -r requirements.txt

## Usage

Run the Brownian dynamics simulation with default test parameters:

    python run_simulation.py

The file `run_simulation.py` calls the `main()` function defined in
`src/simulation.py` and passes a set of example parameters for testing
and demonstration.

## Parameters

The parameters for the main(...) function are:

- packing_fraction (list): List of packing fractions
- file_names (str): Main name of the simulation files
- sigma0 (float): Initial mean particle diameter in most swollen state
- delta_time: Time step size
- N_equ (int): Number of equilibration time steps
- N_dt (int): Number of time steps of the final simulation run
- save_sequence (int): number of time steps after which the particle position etc., are saved

Parameters can be modified directly in `run_simulation.py`.

## Output
The simulation generates a data file for each packing fraction, containing 
the time, number of particles, the dimension of the simulation box, 
the particle positions and diameters.
Files are saved every save_sequence timesteps.

The format of each data file is:

    ITEM: TIMESTEP
    ...
    ITEM: NUMBER OF ATOMS
    ...
    ITEM: BOX BOUNDS
    ... ... xlo xhi
    ... ... ylo yhi
    ... ... zlo zhi
    ITEM: ATOMS id type x y z diameter
    ... ... ... ... ... ...

This format is compatible with the visualization software OVITO, which can be 
used to analyze and animate the particle trajectories.


## License

MIT License