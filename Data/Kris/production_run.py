#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 10:40:04 2023

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
    p.add_argument("-molecule",default="H2O",type=str, help="Molecule: H2O or OH")
    p.add_argument("-noon",default=0,type=int,help="Longitude of Local Noon")
    p.add_argument("-time",default=24,type=int,help="Run time in lunar hours")
    p.add_argument("-dt",default=0.25,type=float,help="Step size in lunar hours")
    p.add_argument("-add",default=100, type=int, help="Amount to add")
    p.add_argument("-num",default=100,type=int,help="Number of particles")
    p.add_argument("-dir",default="../Results/",type=str, help="directory for output")
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

    # Number of added particles
    m = args.add
        
    # Assign directory
    directory= (args.dir)
    
    #print(molecule, local_noon, t, dt, n)

    # ---------------------- RUN THE MODEL
    # establish particles
    particles = np.zeros((n, 3)) # latitude, longitude,  tod
    particles[:, 0] = np.deg2rad(random.choices(range(-90, 90), k=n)) # latitude in degrees
    particles[:, 1] = np.deg2rad(random.choices(range(0, 360), k=n)) # longitude in degrees
    particles[:, 2] = (12+(((np.rad2deg(particles[:, 1]-local_noon))*24)/360))%24 # tod, based on where local noon is

    # run
    results = np.zeros((n+m*int(t/dt), 8, int(t/dt)+1))

    # start timer
    st = time.time()

    # Run model for n particles, 1 lunar day time step 1/2 hr (lunar)
    results = pr.Model_MonteCarlo_produce(particles, dt, t, m, n, local_noon, molecule)

    #print('Total simulation time: %2.1f'%(time.time() - st))
    #print('Lunar time step: %3.2e'%(pr.sec_per_hour_M*dt))

    fheader = "latitude, longitude, time of day, temperature, condition, tot time/step, hops per timestep, distance/step"

    #loc = '/Users/laferrierek/Box Sync/Desktop/Research/Moon_Transport/Writing/Proposals/'
    # write file to .npy, .fits, .txt and .dat
    filename = directory + 'Production_a' + str(m) +'_Smooth_p' +str(n) + '_t' + str(int(t/dt)) +'.npy'
    
    np.save(filename, results, allow_pickle=True)    


if __name__ =='__main__':
    main()

# establish particles
"""
## establish weights
weight_cos = np.linspace(-np.pi/2, np.pi/2, 180)
weight_lon = np.sin(np.linspace(0, 2*np.pi, 360)-np.pi/2)
mask = weight_lon < 0
weight_lon[mask] = 0

particles = np.zeros((n, 3)) # latitude, longitude,  tod
particles[:, 0] = np.deg2rad(random.choices(range(-90, 90), weights=np.cos(weight_cos), k=n)) # latitude in degrees
particles[:, 1] = np.deg2rad(random.choices(range(0, 360), weights=weight_lon, k=n)) # longitude in degrees
particles[:, 2] = (12+(((np.rad2deg(particles[:, 1]-local_noon))*24)/360))%24 # tod, based on where local noon is
"""



