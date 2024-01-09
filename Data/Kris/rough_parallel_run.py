#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 14:36:57 2023

@author: laferrierek
"""

# parallel runs
## if 1000 particles takes ~ 10 hours, needs 3 cores aka 5 GB to run
# we have 64 cores right now (~100 GB)
# can do about 20 runs at a time. 

# 20 runs of 1000 particles each
# total of 20,000 particles over 10 hours

# not sure the space needed for 10,000 particles. that would take 100 hours or 5 days


# this will be the code to run particle loops

# This is for a smooth moon
import numpy as np
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
    p.add_argument("-segment", default = 0, type=int, help="segment (which run of 10?)")
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
        
    # Assign directory
    directory = (args.dirc)
    
    # Assign segment name
    segment = args.segment - 1

    # Load in LOLA RMS roughness maps
    wave = args.scale
    loc = '/home/klaferri/Desktop/Research/Lunar_diurnal_migration/Data/Kris/Roughness/'
    filename = 'MAS_{0:2d}'.format(wave)+'M_16.IMG' # this is on a 57 meter scale. 

    # set width and height - this is true for all three
    w, h = 2880, 5760 

    with open(loc+filename, 'rb') as f: 
        omegas_scaled = np.fromfile(f, dtype=np.int16).reshape(w, h)

    # scaled back to din degrees    
    omegas = omegas_scaled*0.0015 + 45    

    omegaT = omegas.T
    
    # ---------------------- RUN THE MODEL
    # establish particles
    filename = directory + 'Smooth_input_p' + str(n) + "_" + molecule + ".txt"

    particles = np.loadtxt(filename, delimiter =',', skiprows = 1)

    sub = int(np.size(particles[:, 0])/100)
    
    particles = particles[sub*segment:sub*segment+sub, :]
    print("seg:", segment, "start end:", sub*segment, sub*segment+sub)
    # run
    results = np.zeros((sub, 8, int(t/dt)+1))

    # start timer
    st = time.time()

    # Run model for n particles, 1 lunar day time step 1/2 hr (lunar)
    for i in range(0, sub, 1):
        results[i, :, :] = pr.Model_MonteCarlo_Rough(particles[i, :], dt, t, local_noon, molecule, omegaT)
        if i % 10 ==0:
            sys.stderr.flush()  
            print('particle i: %2.0f; time t: %2.0f'%(i, time.time()-st), file=sys.stderr)
            sys.stderr.flush()


    #fheader = "latitude, longitude, time of day, temperature, condition, tot time/step, hops per timestep, distance/step"

    # save as .npy file
    filename = directory + 'Rough_p' +str(n) + '_t' + str(int(t/dt)) + '_' + molecule + '_i' + str(int(segment)) + '.npy'
    
    np.save(filename, results, allow_pickle=True)    


if __name__ =='__main__':
    main()
