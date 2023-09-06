#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 09:37:08 2023

@author: laferrierek
"""

# base run
# This notebook runs a test code
import numpy as np
import random 
import sys
import argparse

# load in custom library
import lunar_processes as pr

# for testing time
import time

def main():
    p=argparse.ArgumentParser(description='Parse inputs for transport')
    p.add_argument("-molecule",default="H2O",type=str,help="Molecule: H2O or OH")
    p.add_argument("-scale", default=57, type=int, help="Roughness scale: 57, 225, 560 m")
    p.add_argument("-noon",default=0,type=int,help="Longitude of Local Noon")
    p.add_argument("-time",default=24,type=int,help="Run time in lunar hours")
    p.add_argument("-dt",default=0.25,type=float,help="Step size in lunar hours")
    p.add_argument("-num",default=100,type=int,help="Number of particles")
    p.add_argument("-dirc",default="../Results/",type=str, help="directory for output")
    args=p.parse_args()
    
    
    # Establish run parameters
    # Establish molecule
    molecule = args.molecule
    
    # Initial longitude of noon
    local_noon = args.noon

    # Run time in lunar hours
    t = args.time

    # Size of time step in lunar hours
    dt = args.dt

    # Number of particles
    n = args.num
    
    # Wavelength of roughness
    wave = args.scale
    
    # Load in LOLA RMS roughness mpas
    loc = '/Users/laferrierek/Box Sync/Desktop/Research/Moon_Transport/Codes/Data/'
    filename = 'MAS_{0:2d}'.format(wave)+'M_16.img' # this is on a 57 meter scale. 

    # set width and height - this is true for all three
    w, h = 2880, 5760 

    with open(loc+filename, 'rb') as f: 
        omegas_scaled = np.fromfile(f, dtype=np.int16).reshape(w, h)

    # scaled back to din degrees    
    omegas = omegas_scaled*0.0015 + 45    

    omegaT = omegas.T
            
    # Assign directory
    directory= (args.dirc)
    
    # ---------------------- RUN THE MODEL
    # establish particles
    particles = np.zeros((n, 3)) # latitude, longitude,  tod
    particles[:, 0] = np.deg2rad(random.choices(range(-90, 90), k=n)) # latitude in degrees
    particles[:, 1] = np.deg2rad(random.choices(range(0, 360), k=n)) # longitude in degrees
    particles[:, 2] = (12+(((np.rad2deg(particles[:, 1]-local_noon))*24)/360))%24 # tod, based on where local noon is

    # run
    results = np.zeros((n, 8, int(t/dt)))

    # start timer
    st = time.time()

    # Run model for n particles, 1 lunar day time step 1/2 hr (lunar)
    for i in range(0, n, 1):
        results[i, :, :] = pr.Model_MonteCarlo_Rough(particles[i, :], dt, t, local_noon, molecule, omegaT)
        if i % 10 ==0:
            sys.stderr.flush()  
            print('particle i: %2.0f; time t: %2.0f'%(i, time.time()-st), file=sys.stderr)
            sys.stderr.flush()

    fheader = "latitude, longitude, time of day, temperature, condition, tot time/step, hops per timestep, distance/step"

    #loc = '/Users/laferrierek/Box Sync/Desktop/Research/Moon_Transport/Writing/Proposals/'
    # write file to .npy, .fits, .txt and .dat
    filename = directory + 'Rough_p' +str(n) + '_t' + str(int(t*dt)) +'.npy'

    np.save(filename, results, allow_pickle=True)    


if __name__ =='__main__':
    main()

