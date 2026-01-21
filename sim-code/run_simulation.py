#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 15:51:04 2025

@author: jp
"""



import time

from simulation import main
###############################################################################
#Test parameters for main()                                                   #
###############################################################################

packing_fraction = [0.10, 1.00]
test_name = r"test-run"
test_sigma0 = 44.48161863
    #simulation settings
test_delta_time = 5e-3#time step
test_N_equ = int(1e5) #equilibration time steps
test_N_dt = int(1e5) #number of time steps
test_save_sequence = 1000


###############################################################################
#Test call of the main function for three packing fractions                   #
###############################################################################


start = time.time()
main(packing_fraction, test_name, test_sigma0, test_delta_time, test_N_equ, test_N_dt, test_save_sequence)
end = time.time()

ct = end - start
hours = int(ct // 3600)
minutes = int((ct % 3600) / 60)
seconds = int(ct % 60)
print(f"Execution time: {hours}h; {minutes}min; {seconds: .2f}s")