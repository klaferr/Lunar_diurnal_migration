#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 15:27:55 2023

@author: laferrierek
"""

# this will be the code to create_particles.py
# This is for a smooth moon
import numpy as np
import random 
import argparse

def main():
    p=argparse.ArgumentParser(description='Parse inputs for transport')
    p.add_argument("-molecule",default="H2O",type=str,help="Molecule: H2O or OH")
    p.add_argument("-noon",default=0,type=int,help="Longitude of Local Noon")
    p.add_argument("-num",default=100,type=int,help="Number of particles")
    p.add_argument("-dirc",default="../Results/",type=str, help="directory for output")
    args=p.parse_args()
    
    # Establish run parameters
    # Establish molecule
    molecule = args.molecule

    # Initial longitude of noon
    local_noon = args.noon

    # Number of particles
    n = args.num
        
    # Assign directory
    directory = (args.dirc)
    
    # ---------------------- RUN THE MODEL
    # establish particles
    particles = np.zeros((n, 3)) # latitude, longitude,  tod
    particles[:, 0] = np.deg2rad(random.choices(range(-90, 90), k=n)) # latitude in degrees
    particles[:, 1] = np.deg2rad(random.choices(range(0, 360), k=n)) # longitude in degrees
    particles[:, 2] = (12+(((np.rad2deg(particles[:, 1]-local_noon))*24)/360))%24 # tod, based on where local noon is

    fheader = "latitude, longitude, time of day"

    filename = directory + 'Smooth_input_p' +str(n) + "_" + molecule +'.txt'
    
    np.savetxt(filename, particles, delimiter=',', header=fheader)    


if __name__ =='__main__':
    main()