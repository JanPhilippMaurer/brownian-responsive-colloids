#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 2025

@author: Jan-Philipp Maurer
"""
import numpy as np
from numba import njit

###############################################################################
###                     Psi from a simple gaussian                          ###
###############################################################################
#properti potential function fitted to a quadratic function of gaussian origin
@njit
def psi_gaussian(sigma):
    a0, s0, d0 =  0.2337347, 44.48161863, 2.37600334 
    
    return -np.log(a0) + ((sigma-s0)**2 /d0**2)

#property force calculated from derivative of psi_gaussian(sigma)
@njit
def force_psi_gaussian_d_sigma(sigma):
    a0, s0, d0 =  0.2337347, 44.48161863, 2.37600334 
    
    deri = 2 * (sigma-s0) / d0**2
    return -deri


###############################################################################
###                     Psi from a quadric function                         ###
###############################################################################
#properti potential function fitted to a quartic form
@njit
def psi_quartic(sigma):
    E0,a0,s0 = 3.00017100e-03, 1.71703355e-01, 4.44940168e+01
    b0,b1,b2 = 2.08216596e+01, -8.01055975e-01,  7.93129159e-03
    
    return (E0 + a0*(sigma-s0)**2) * (b0 + b1*sigma, + b2*sigma**2)

#property force calculated from derivative of psi_quartic(sigma)
@njit
def force_psi_quartic_d_sigma(sigma):
    E0,a0,s0 = 3.00017100e-03, 1.71703355e-01, 4.44940168e+01
    b0,b1,b2 = 2.08216596e+01, -8.01055975e-01,  7.93129159e-03
    
    deri1 = (2*a0*(sigma-s0)) * (b0 + b1*sigma + b2*sigma**2)
    deri2 = (E0 + a0*(sigma-s0)**2) * (b1 + 2*b2*sigma)
    deri = deri1 + deri2
    return -deri


###############################################################################
###                     Psi from a FENE like function                       ###
###############################################################################
#properti potential function fitted to a FeNE-like form
@njit
def psi_FENE1(x):
    a, s0, c = -130.27611764, 44.70648452, -0.13860695
    R = 25
    return a*np.log(1-((x-s0)/44)**2)+c
@njit
def psi_FENE2(x):
    a, s0, c = -1.88547357e+02, 4.47066855e+01, -1.41018024e-01
    R = 30
    return a*np.log(1-((x-s0)/44)**2)+c

#property force calculated from derivative of psi_FENE1/2(sigma)
@njit
def force_psi_FENE1_d_sigma(sigma):
    a,s0,c = -130.27611764, 44.70648452, -0.13860695
    R = 25
    deri = a * (2*(sigma-s0)**2/R**2) / (1-((sigma-s0)/R)**2)
    return -deri
@njit
def force_psi_FENE2_d_sigma(sigma):
    a,s0,c = -1.88547357e+02,  4.47066855e+01, -1.41018024e-01
    R = 30
    deri = a * (2*(sigma-s0)**2/R**2) / (1-((sigma-s0)/R)**2)
    return -deri






###############################################################################
###             Phi as a Hertzian of a core/shell colloid                   ###
###############################################################################


@njit
def shell_core_hertzian_phi(r,s_i,s_j,c):
    nu = 0.3
    Y = c / s_i**3
    Yp = c / s_j**3
    A = Y / (1-nu**2)
    Ap = Yp / (1-nu**2)

    Aeff = (A*Ap) / (A+Ap)
    
    s_ij = 0.5*(s_i+s_j)

    epsilon = (8/15) * Aeff * (0.5*s_i+0.5*s_j)**2 * (0.25*s_i*s_j)**(1/2)

    
    pot = np.where(r<s_ij, epsilon*(1-r/s_ij)**(5/2),0.0)
    return pot




@njit
def force_phi_shell_core_hertzian_d_sigma(r,s_i:float, s_j:float):
    """
    Calculates the property force F_sigma comming from the pair-potential phi

    Parameters
    ----------
    r : float
        distance between colloid i and j
    sigma_ij : float
        effective colloid size: 0.5*(sigma_i + sigma_j)

    Returns
    -------
    float
        property force

    """
    #########
    c = 750 #sets potential strength
    #######
    
    f = 0.0
    s_ij = 0.5*(s_i+s_j)
    
    frac = r / s_ij
    d =  1 - frac
    if d > 0:
        nu = 0.3
        Y = c / s_i**3
        Yp = c / s_j**3
        A = Y / (1-nu**2)
        Ap = Yp / (1-nu**2)

        Aeff = (A*Ap) / (A+Ap)
        

        epsilon = (8/15) * Aeff * (0.5*s_i+0.5*s_j)**2 * (0.25*s_i*s_j)**(1/2)
        
        
        d1 = epsilon * (5/4)*(r/s_ij**2)*(1-r/s_ij)**(3/2)
        
        d21 = (8/15) * (Aeff * s_ij**2 * (s_j/8)*(0.25*s_i*s_j)**(-1/2))
        d22 = (8/15) * (Aeff * (0.25*s_i*s_j)**(1/2) * s_ij)
        dAeff = - (3/s_i) * ((A*Ap**2) / (A+Ap)**2)
        d23 = (8/15) * (dAeff * (0.25*s_i*s_j)**(1/2) * s_ij**2)
        
        d2 = (d21 + d22 + d23) * (1-r/s_ij)**(5/2)
        
        f = d1 + d2 
    return -f





@njit
def force_phi_shell_core_hertzian_dr(vector:list,r:float,s_i:float, s_j:float):
    """
    calculates the translation force between two colloids i and j

    Parameters
    ----------
    vector : list
        distance vector between colloid i and j
    r : float
        distance between colloid i and j
    sigma_ij : float
        effective colloid size: 0.5*(sigma_i + sigma_j)

    Returns
    -------
    f_vector : list
        calculated force vector

    """
    #########
    c = 750 #sets potential strength
    #######
    
   
    s_ij = 0.5*(s_i+s_j)

    f_vector = np.zeros(3)
    
    
    frac = r / s_ij
    d = 1 - frac
    if d > 0:
        nu = 0.3
        Y = c / s_i**3
        Yp = c / s_j**3
        A = Y / (1-nu**2)
        Ap = Yp / (1-nu**2)

        Aeff = (A*Ap) / (A+Ap)

        epsilon = (8/15) * Aeff * (0.5*s_i+0.5*s_j)**2 * (0.25*s_i*s_j)**(1/2)
        
        pre = (epsilon / (s_ij * r))
        pre1 = 2.5 * pre * d**1.5
        
        for d in range(3):
            f_vector[d] += vector[d] * pre1
    return f_vector

