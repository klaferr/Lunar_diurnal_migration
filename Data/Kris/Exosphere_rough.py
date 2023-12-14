#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 11:00:03 2023

@author: laferrierek

this is for testing exosphere particles
## run same particle many times, record each hop height, distance, time etc


"""

# import libraries
import numpy as np
import random 
import sys
import argparse

from scipy import interpolate

# load in custom library
import lunar_processes as pr
from lunar_processes import G, moonM, moonR
# for testing time
import time

#%% functions
def ballistic_tof(vm, phi):
    # Butler 1997

    voy = vm*np.sin(phi)             # m/s
    
    g0 = G*moonM/moonR**2                # m/s2
    hmax = (moonR * voy**2)/ (moonR*2*g0 - voy**2)   # m ?
    
    # now find the time of flight
    v = moonR + hmax           # give this a new lettter
    g = (G*moonM)/((moonR)**2) #(G*moonM)/((moonR+hmax)**2)
    a = voy**2 * moonR
    b = voy**2 - 2*g*moonR
    
    u = a+b*hmax  
    l = a - b*moonR
    p = (2*b*hmax+a+(b*moonR))/l
    
    t =  ((v/np.abs(v))*(np.sqrt(u*v)/b + l/(2*b)*((1/np.sqrt(-b))*np.arcsin(p)))) # at hmax
    
    v = moonR 
    g = G*moonM/moonR**2
    a = voy**2 * moonR
    b = voy**2 - 2*g*moonR
    
    u = a+b*0
    l = a - b*moonR
    p = (2*b*0+a+(b*moonR))/l
    t0 =  ((v/np.abs(v))*(np.sqrt(u*v)/b + l/(2*b)*((1/np.sqrt(-b))*np.arcsin(p)))) # at0
    
    time_of_flight = 2*(t-t0)
    
    hmax = (moonR*voy**2)/(2*moonR*g-voy**2)

    # if hmax > 6.15*10**7: #equation: a*(moonM/3*earthM)**(1/3); where a = 384400 km, 
        # raise Exception("exceed's hill sphere!")
        # print("hmax:", hmax)
    return time_of_flight, hmax 

# outputting height with smooth moon model
def nan_tof(vm, phi):
    vtest = np.arange(vm-100, vm+100, 1)
    vm_loc = np.argwhere(vtest == vm)[0][0]
    time, height = ballistic_tof(vtest, phi)
    
    mask = np.isnan(time)
    spline = interpolate.InterpolatedUnivariateSpline(vtest[~mask], time[~mask])
    time_new = spline(vtest)
    
    voy = vm*np.sin(phi) 
    g = (pr.G*pr.moonM)/((pr.moonR)**2)
    height = (pr.moonR*voy**2)/(2*moonR*g-voy**2)
    return time_new[vm_loc], height


def ballistic(temp, i_lat, i_long, i_tod, direction, launch, pMass, vel_dist):

    s1 = pr.maxwell_boltz_dist(pr.vel_dist, pMass, temp)

    particle_v = random.choices(vel_dist, weights=s1/np.nanmax(s1))[0]  
    cond = False
    if particle_v > pr.vesc:
       cond = True

    dist_m = pr.ballistic_distance(launch, moonR, particle_v, moonM)
    f_lat, f_long = pr.landing_loc(i_lat, i_long, dist_m, moonR, direction)
    f_tof, height = ballistic_tof(particle_v, launch)
    
    if np.isnan(f_tof) == True:
        f_tof, height = nan_tof(particle_v, launch)
    
    f_tod = pr.time_of_day(i_long, f_long, i_tod, f_tof)
    return np.array([f_lat, f_long, f_tod]), f_tof, height, dist_m, cond


def exosphere_multiple(particle, tt, dt, t, local_noon, molecule):
    if molecule == "H2O":
        R_bar = pr.R/(pr.m_H2O/1000)
        pMass = pr.mass_H2O
        sigma_Sputtering = 1/pr.sput_rate_S23_H2O
        photo_lifespan =  1/pr.photo_rate_S14_H2O
    elif molecule == "OH":
        R_bar = pr.R/(pr.m_H2O/1000)
        pMass = pr.mass_H2O
        sigma_Sputtering = 1/pr.sput_rate_G19_OH
        photo_lifespan =  1/pr.photo_rate_S23_OH
        
    else:
        raise Exception("Must be OH or H2O")
    
    out = np.zeros((9, tt))*np.nan
    
    # if exists = True, set 1. Else, set 0
    tot_time = 0
    out[0:3, 0] = particle
    out[4, 0] = False
    conda = False
    condb = False
    
    lunar_dt = pr.sec_per_hour_M*dt
        
    # Run model for 1 particles, up to 1 lunar day with however many timesteps we need
    for i in range(0, tt-1, 1):
        out[3, i], num = pr.DivinerT(out[0, i], out[1, i], out[2, i], pr.data)

        # define how long it sits for
        tau_surf = pr.surftime(R_bar, out[3, i], pMass)  

        # if it sits for a timestep, test lost, then move to next timestep
        if tau_surf >= lunar_dt:
            #print('Particle sits longer than a lunar time step')

            conda = pr.loss(sigma_Sputtering, photo_lifespan, lunar_dt, out[2, i], cosi)
            if conda == True:
                out[4, i] = conda
                #print('lost, long sit')
                break
            else:
                out[0, i+1] = out[0, i]
                out[1, i+1] = out[1, i]

                # rotate Moon timestep 
                local_noon += (360/(t/dt)) # degrees
                out[2, i+1] = (12 + (np.rad2deg(out[1, i+1])+local_noon)*(24/360))%24   

            out[4, i] = conda
            # 5: launch angle, 6:height, 7:velcoity, 8:time of flight
            out[5, i] = 0
            out[6, i] = 0
            out[7, i] = 0
            out[8, i] = 0
            tot_time += lunar_dt

        else:

            # test lost from sit
            conda = pr.loss(sigma_Sputtering, photo_lifespan, tau_surf, out[2, i])
            if conda == True:
                out[4, i+1] = conda
                #print('lost, short sit')
                break

            # now jump
            direction, out[5, i] = pr.random_vals()   

            s1 = pr.maxwell_boltz_dist(pr.vel_dist, pMass, out[3, i])
            out[7, i] = random.choices(pr.vel_dist, weights=s1/np.nanmax(s1))[0]  

            if out[7, i] > pr.vesc:
                raise Exception("particle escapes - Jeans")

            out[8, i], out[6, i] = ballistic_tof(out[7, i], out[5, i])
            if np.isnan(out[8, i]) == True:
                out[8, i] = pr.nan_tof(out[7, i], out[5, i])
                print("Issue in TOF equation")
            if out[6, i] > 6.15*10**7:
                raise Exception("particle escapes - Hills")

            dist_m = pr.ballistic_distance(out[5, i], moonR, out[7, i], moonM)
            out[0, i+1], out[1, i+1] = pr.landing_loc(out[0, i], out[1, i], dist_m, moonR, direction)
            out[2, i+1] = pr.time_of_day(out[1, i],  out[1, i+1], out[2, i], out[8, i])

            # is it detroyed in the jump?
            condb = pr.loss(sigma_Sputtering, photo_lifespan, out[8, i], out[2, i])
            if condb == True:
                out[4, i+1] = condb
                #print(cosi, out[0, i])
                #print('lost, jump')
                break

            else:
                out[3, i+1], num = pr.DivinerT(out[0, i+1], out[1, i+1], out[2, i+1], pr.data)
                tot_time += tau_surf + out[8, i]
                
            if tot_time >= 24*pr.sec_per_hour_M:
                #print("tot time exceeds 24 lunar hours")
                break

    #print('Particle experiences (s): %2.1f'%tot_time)    
    #print('Total time passed in sim (lunar hours):%3.2f'%(tot_time/pr.sec_per_hour_M))

    return out

def exosphere_multiple_rough(particle, tt, dt, t, local_noon, molecule, omegaT):
    if molecule == "H2O":
        R_bar = pr.R/(pr.m_H2O/1000)
        pMass = pr.mass_H2O
        sigma_Sputtering = 1/pr.sput_rate_S23_H2O
        photo_lifespan =  1/pr.photo_rate_S14_H2O
    elif molecule == "OH":
        R_bar = pr.R/(pr.m_H2O/1000)
        pMass = pr.mass_H2O
        sigma_Sputtering = 1/pr.sput_rate_G19_OH
        photo_lifespan =  1/pr.photo_rate_S23_OH
        
    else:
        raise Exception("Must be OH or H2O")
    out = np.zeros((9, tt))*np.nan
    
    # if exists = True, set 1. Else, set 0
    tot_time = 0
    out[0:3, 0] = particle
    out[4, 0] = False
    conda = False
    condb = False
    
    lunar_dt = pr.sec_per_hour_M*dt
        
    # Run model for 1 particles, up to 1 lunar day with however many timesteps we need
    for i in range(0, tt-1, 1):
        out[3, i], alpha = pr.roughT(out[0, i], out[1, i], out[2, i], pr.data, omegaT)

        # define how long it sits for
        tau_surf = pr.surftime(R_bar, out[3, i], pMass)  

        cosi = pr.cosi_rough(out[0, i], 0, np.deg2rad(15*out[2, i])-np.pi, 1/(np.pi*2), alpha)

        # if it sits for a timestep, test lost, then move to next timestep
        if tau_surf >= lunar_dt:
            #print('Particle sits longer than a lunar time step')

            conda = pr.loss(sigma_Sputtering, photo_lifespan, lunar_dt, out[2, i], cosi)
            if conda == True:
                out[4, i] = conda
                #print('lost, long sit')
                break
            else:
                out[0, i+1] = out[0, i]
                out[1, i+1] = out[1, i]

                # rotate Moon timestep 
                local_noon += (360/(t/dt)) # degrees
                out[2, i+1] = (12 + (np.rad2deg(out[1, i+1])+local_noon)*(24/360))%24   

            out[4, i] = conda
            # 5: launch angle, 6:height, 7:velcoity, 8:time of flight
            out[5, i] = 0
            out[6, i] = 0
            out[7, i] = 0
            out[8, i] = 0
            tot_time += lunar_dt

        else:

            # test lost from sit
            conda = pr.loss(sigma_Sputtering, photo_lifespan, tau_surf, out[2, i], cosi)
            if conda == True:
                out[4, i+1] = conda
                #print('lost, short sit')
                break

            # now jump
            direction, out[5, i] = pr.random_vals()   

            s1 = pr.maxwell_boltz_dist(pr.vel_dist, pMass, out[3, i])
            out[7, i] = random.choices(pr.vel_dist, weights=s1/np.nanmax(s1))[0]  

            if out[7, i] > pr.vesc:
                raise Exception("particle escapes - Jeans")

            out[8, i], out[6, i] = ballistic_tof(out[7, i], out[5, i])
            if np.isnan(out[8, i]) == True:
                out[8, i] = pr.nan_tof(out[7, i], out[5, i])
                print("Issue in TOF equation")
            if out[6, i] > 6.15*10**7:
                raise Exception("particle escapes - Hills")

            dist_m = pr.ballistic_distance(out[5, i], moonR, out[7, i], moonM)
            out[0, i+1], out[1, i+1] = pr.landing_loc(out[0, i], out[1, i], dist_m, moonR, direction)
            out[2, i+1] = pr.time_of_day(out[1, i],  out[1, i+1], out[2, i], out[8, i])

            # is it detroyed in the jump?
            condb = pr.loss(sigma_Sputtering, photo_lifespan, out[8, i], out[2, i], cosi)
            if condb == True:
                out[4, i+1] = condb
                #print(cosi, out[0, i])
                #print('lost, jump')
                break

            else:
                out[3, i+1], alpha = pr.roughT(out[0, i+1], out[1, i+1], out[2, i+1], pr.data, omegaT)
                tot_time += tau_surf + out[8, i]
                
            if tot_time >= 24*pr.sec_per_hour_M:
                #print("tot time exceeds 24 lunar hours")
                break

    #print('Particle experiences (s): %2.1f'%tot_time)    
    #print('Total time passed in sim (lunar hours):%3.2f'%(tot_time/pr.sec_per_hour_M))

    return out



def main():
    p=argparse.ArgumentParser(description='Parse inputs for transport')
    p.add_argument("-molecule",default="H2O",type=str,help="Molecule: H2O or OH")
    p.add_argument("-scale", default=57, type=int, help="Roughness scale: 57, 225, 560m")
    p.add_argument("-noon",default=0,type=int,help="Longitude of Local Noon")
    p.add_argument("-time",default=24,type=int,help="Run time in lunar hours")
    p.add_argument("-dt",default=0.25,type=float,help="Step size in lunar hours")
    p.add_argument("-tt",default=250,type=int,help="Jumps per step max")
    p.add_argument("-num",default=100,type=int,help="Number of particles")
    p.add_argument("-segment", default = 0, type=int, help="segment (which run of 10?)")
    p.add_argument("-dirc",default="../Results/",type=str, help="directory for output")
    args=p.parse_args()
    
    # Establish run parameters
    # Establish molecule
    molecule = args.molecule
    
    # Initial longitude of noon
    local_noon = args.noon
    local_noon_reset = local_noon

    # Run time in lunar hours
    t = args.time

    # Size of time step in lunar hours
    dt = args.dt

    # Number of particles
    nn = args.num
    
    # timesteps
    tt = args.tt
        
    # Assign directory
    directory = (args.dirc)
    
    # Assign segment name
    segment = args.segment

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
    
    # ---------------------- RUN THE MODEL ---------------------- #
    # establish particles
    filename = directory + 'Smooth_input_p' + str(nn) + "_" + molecule + ".txt"
    particles = np.loadtxt(filename, delimiter =',', skiprows = 1)

    sub = int(np.size(particles[:, 0])/100)

    particles = particles[sub*segment:sub*segment+sub, :]

    # empty bins - # lat, long, tod, temp, exists, launch angle, height, velcoity, time of flight
    results = np.zeros((sub, 9, tt))*np.nan

    # start timer
    st = time.time()

    # Run model for n particles, 1 lunar day time step 1/2 hr (lunar)
    for n in range(0, sub, 1):
        local_noon = local_noon_reset
        results[n, :, :] = exosphere_multiple_rough(particles[n, :], tt, dt, t, local_noon, molecule, omegaT)
        if n % 10 ==0:
            sys.stderr.flush()  
            print('particle n: %2.0f; time t: %2.0f'%(n, time.time()-st), file=sys.stderr)
            sys.stderr.flush()


    #fheader = "latitude, longitude, time of day, temperature, condition, tot time/step, hops per timestep, distance/step"

    # save as .npy file
    filename = directory + 'Exosphere_rough_p' +str(nn) + '_t' + str(int(tt)) + '_' + molecule + '_i' + str(int(segment)) + '.npy'
    
    np.save(filename, results, allow_pickle=True)    


if __name__ =='__main__':
    main()


#%% 

