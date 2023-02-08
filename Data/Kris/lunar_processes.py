#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 12:37:28 2023

@author: laferrierek
"""

#%% import libraries
import numpy as np
from pathlib import Path
import random
from scipy import interpolate

#%% Define values for inputs
# Constants
Avo = 6.0221*10**(23)               # 1/ mol
G = 6.6743*10**(-11)                # m3/kg/s2
kb = 1.38064*10**(-23)              # J/K
R = 8.3145                          # J/mol/K
vel_dist = np.arange(0, 3000, 1)

# Molecules
m_H2O= 18.01528                     # g/ mol
m_OH = 17.008                       # g/mol
mass = (m_H2O/1000) / Avo           # kg/mol / (1/mol)
density_H2O = 997

# Molecule - Water
triple_T = 273.1575                 # K
triple_P = 611.657                  # Pa 
Lc =  2.834*10**6                   # J/kg - is this wrong
Q = 51058                           # J/mol

# Earth
sec_per_day_E = 24*60*60
sec_per_hour_E = 24*60
Gyr_E = (10**9 * 365.2*sec_per_day_E)
rEarth = 1                          # AU

# Lunar Values
# nasa fact sheet - sideread roatation == revolution period
# day == synodic period? need to check.lunation
rotation_Edays = 27.3217 # Synodic, lunation
rev_Edays = 29.53 # Sideread

sec_per_day_M = sec_per_day_E * 29.53
sec_per_hour_M = sec_per_day_M/24
sec_per_min_M = sec_per_hour_M/60

moonM = 7.3477*10**22               # kg
moonR = 1738.1*10**3                # m, equatorial radiusm
moonR_polar = 1736*10**3            # m, polar radius
avg_Bond = 0.123                    # Steckloff et al. (2022)

vesc = np.sqrt((2*G*moonM)/moonR)   # m/s

#%% Processes rates

## Sources: Rates of supply
# Solar wind - Hurley et al. (2017)
Solar_wind_flux = 5*10**(11)                        # kg in 1 Gyr
Solar_wind_density = 1.38*10**(-2)                  # kg/Ga/m^2
Solar_wind_diurnal = Solar_wind_flux/(Gyr_E)        # kg/s

# Comet flux of water molecules 
Comet_flux = 2.5*10**15                             # kg/Ga
Comet_density = 66                                  # kg/Ga/m^2
Comet_diurnal = Comet_flux/(Gyr_E)

## Residence timescale

# Photolysis -options
photo_rate_S14 = 1.26*10**(-5)                      # 1/s
photo_lifespan_S14 = 1/photo_rate_S14               # s

photo_lifespan_H92 = 5*10**4                        # s

photo_lifespan_G19 = [1.5*10**5, 5*10**6]           # s
photo_probabiility_inflight_G19 = 0.02

photo_lifespan_S22 = 25*sec_per_hour_E

photo_rate_H92 = 1.20*10**(-5) # 1/s
#? = np.cos(i)/lifespan_G19

# Diffusion
diff_lifespan = [10**(-8), 10**(-6)]
diff_activation_energy = 0.8 #eV

# Sputtering
sput_rate_G19 = 1.3*10**(-10)           # 1/s
sput_lifespan_G19 = 1/sput_rate_G19
F_solarwind_proton_S01 = 3*10**12       # m^-2 s^-1

#h_imp = 10**(-7)    # implantation depth Farrell et al. (2015), m
#Y_0 = 0.1           # total sputteirng yield Wurz et al. (2007), unitless
#C_0 = 2.3*10**23    # atomic abundance for anorthite Vasavada 1999, /m^3

# exosphere
exo_lifespan = 23 # hr

# h2o on h2o, Prem et al. 2015
deltaH = 6.6*10**(-20) # J, binding energy for H2O on H2O matrix
v0 = 2*10**(12) #1/s, lattice vibrational frequency of H2O within H2O matrix)



#%% Transport functions
# Orbits
def haversine(lon1, lat1, lon2, lat2, radius):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    return c * radius   #km

def semi_major(r, v, M):
    # 
    # Input: radius (m), velocity (m/s), mass (kg)
    # Output: semi-major axis (m)
    return (1/((2/r)-(v**2/(G*M))))

def ecc(r, a, phi):
    #
    # Input: radius (m), semi_major (m), phi (radians)
    # Output: eccentricity (?)
    return np.sqrt(1- (r*(2*a-r)*(np.sin(phi))**2)/a**2)

def omega(a, E, r):
    #
    # Input: semi_major (m), eccentricity (?), radius (m)
    # Output: true anomaly (?)
    return np.arccos((a-a*E**2-r)/(r*E))
    
# Ballistic Transport:
def maxwell_boltz_dist(vel, mass, Temp):
    # For a given T and mass of a molecule
    # need mass of a molecule in kg!
    # Input: Velocity array, mass, and Temperature
    # Output: a probabilty distribution for the velocity array
    return (4*np.pi)*(mass/(2*np.pi*kb*Temp))**(3/2)*(vel**2)*(np.exp(-((mass*vel**2)/(2*kb*Temp))))

def ballistic_distance(phi_rad, radius, vel, mass):
    # From Sori et al. (2017)
    # mass: mass of the MOON, kg
    a = semi_major(radius, vel, mass)
    E = ecc(radius, a, phi_rad)
    w = omega(a, E, radius)
    dball = (2*np.pi-2*w)*radius
    return dball

def time_of_day(long1, long2, time1, tof):
    # for a non-rotating moon. assumes time in flight << 1 hour. 
    time_i = time1+ tof/sec_per_hour_M
    Lhour_per_deg = 24/360
    delta_lon = np.rad2deg((long2-long1))
    new_time = time_i + Lhour_per_deg*delta_lon
    #if time_i > sec_per_hour_M:
    #    print("Moon rotates a significant portion")
    return new_time%24

def nan_tof(vm, phi):
    vtest = np.arange(vm-100, vm+100, 1)
    vm_loc = np.argwhere(vtest == vm)[0][0]
    time = ballistic_tof(vtest, phi)
    
    mask = np.isnan(time)
    spline = interpolate.InterpolatedUnivariateSpline(vtest[~mask], time[~mask])
    time_new = spline(vtest)
    return time_new[vm_loc]

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
    return time_of_flight 

def ballistic(temp, i_lat, i_long, i_tod, direction, launch, pMass, vel_dist):
    # note, as some point, we should take the weighted average, rather than max. 
    s1 = maxwell_boltz_dist(vel_dist, pMass, temp)
    #particle_v =  vel_dist[int(np.argwhere(np.nanmax(s1)==s1))]

    particle_v = random.choices(vel_dist, weights=s1/np.nanmax(s1))[0]  
    cond = False
    if particle_v > vesc:
       cond = True
    #print(particle_v, launch)

    dist_m = ballistic_distance(launch, moonR, particle_v, moonM)
    f_lat, f_long = landing_loc(i_lat, i_long, dist_m, moonR, direction)
    f_tof = ballistic_tof(particle_v, launch)
    if np.isnan(f_tof) == True:
        f_tof = nan_tof(particle_v, launch)
    f_tod = time_of_day(i_long, f_long, i_tod, f_tof)
    return np.array([f_lat, f_long, f_tod]), f_tof, dist_m, cond

def landing_loc(in_lat, in_long, distance, radius, travel_dir):
    #given a distance and angle, calculate...
    # from Menten+ 2022
    # output in radians, input in radians
    lat2 = np.arcsin(np.sin(in_lat)*np.cos(distance/radius)+np.cos(in_lat)*np.sin(distance/radius)*np.cos(travel_dir))
    long2 = in_long + np.arctan2(np.sin(travel_dir)*np.sin(distance/radius)*np.cos(in_lat), np.cos(distance/radius)-np.sin(in_lat)*np.sin(lat2))
    return lat2, long2%(2*np.pi)

#%% Loss functions
## Sputtering
def sputtering_prob(tau, tof):
    prob = tof/tau
    test = random.randint(0, 99)
    happens = random.sample(range(0,100), round(prob*100))
    if test in happens:
        sputter = True
    else:
        sputter = False
    return sputter

def sputtering(F_sw, h_imp, Y_0, C_0):
    # Grumpe et al. 2019 - following eq 4
    # F_sw = solar wind
    # h_imp = 10**(-7) # m - implantation depth
    # Y_O = total sputtering yield for O ~ 0.1 
    # C_O = oxygen number density ~2.3*10^28 /m^3 for anorthite
    
    # this is equal to 1.3*10**(-10) /s
    return (F_sw/h_imp)*(Y_0/C_0)

# Photodissociation
def photodestruction_inflight_prob(tau, tof):
    # this is equiv. to 1-e^(-t/tau), where if happens < prob, it is lost
    # here, it is if happens > prob, it is lost
    probability = np.exp(-tof/tau)
    happens = random.random()
    if happens >= probability:
        destroy = True
    else:
        destroy = False
    return destroy

# Sublimation
def clapyeron(triple_pressure, triple_T, R_bar, Lc, T):
    # Pascal
    # Lc/R_bar or Q/R, where R_bar = R/mass_molecule
    return triple_pressure*np.exp( (Lc/R_bar) * ((1/triple_T) - (1/T)))

def sublimation_depth(mu, delta, z, E):
    # Assumes E(z) can be calculated for a T at z. 
    return (1/Avo)*mu*delta*E/(2*z)      # kg/m2/s

def sublimation_surf(P, T, mu):
    # Siegler et al. 2015
    # inputs: 
    # P = saturation vapor pressure - kg/m/s2
    # mu is molecular weight - input is ~ 0.018, kg/mol
    # R is J/K/mol
    # Output is: kg/m2/s
    ##return P/np.sqrt(2*np.pi*R*T/mu)     # output units are kg/m2/s

    # Schorghofer and Taylor 2007
    # Inputs:
    # P = sat vap rpessure kg/m/s2
    # mu is mass of a molecule - input is 0.018 kg/mol
    # kb is J/K/mol 8.314
    # output units are #/m2/s
    ##return P/np.sqrt(2*np.pi*kb*T*mu)

    # Schorghofer et al 2022
    # Inputs:
    # P - sat vap pressure - kg/m/s2
    # m - mass of a molecule - kg/mol
    # R = J/K/mol (paper has kb, but in just of R. )
    # mu is mass of a molecule. input is ~0.018/Avo
    # output units are kg/m2/s
    return P*np.sqrt(mu/(2*np.pi*R*T))

# Diffusion
#def diffusion():
#    return

# desorption
def desorption_R21(T, Ed):
    # From Sarantos and Tsavachidis 2021
    # Ed is actiation energy in eV, ranges from 0.6-1.9 eV # poston
    A = 10**13 # 1/s
    kb_eV = kb/(1.60218*10**(-19))
    R_des = A*np.exp(-Ed/(kb_eV*T))
    tau = 1/R_des # this is simplified, to assume mean residence time
    return tau

def desorp_prob(tau, tof):
    # this is equiv. to 1-e^(-t/tau), where if happens < prob, it is lost
    # here, it is if happens > prob, it is lost
    probability = np.exp(-tof/tau)
    happens = random.random()
    if happens >= probability:
        destroy = True
    else:
        destroy = False
    return destroy

def desorption_F15(T, U, h):
    # From Farrell, Hurley, Zimmerman 2015
    D = 10**(-6) # m^2/s
    kb_eV = kb/(1.60218*10**(-19))
    return h**2/D * np.exp(U/(kb_eV*T))

def absorption(tau):
    # From Sarantos and Tsavachidis 2021
    Rdes = 1/tau
    e = random.uniform(0, 1)
    ta = -np.ln(1-e)/Rdes
    return ta


#%% Temperature from data
def DivinerT(lat_r, long_r, tod, data):
    delta = 0.25*2
    delta_tod = 0.28*2
    lat = np.rad2deg(lat_r)
    long = np.rad2deg(long_r - np.pi) 
    
    mask = (data[:, :, 0] >= long - delta) & (data[:, :, 0] <= long+delta) & (data[:, :, 1] >= lat - delta) & (data[:, :, 1] <= lat+delta)
    data_loc = data[mask, :]
    
    if np.size(data_loc) == 0:
        print(lat, long)
        raise Exception ('Lacking lat or long data')
    
    if tod < delta_tod:
        masktod = (data_loc[:, 2] > (tod - delta_tod)%24 ) | (data_loc[:, 2] < (tod+delta_tod)%24)
    elif tod > 24-delta_tod:
        masktod = (data_loc[:, 2] > (tod - delta_tod)%24 ) | (data_loc[:, 2] < (tod+delta_tod)%24)
    else:            
        masktod = (data_loc[:, 2] >= (tod - delta_tod)) & (data_loc[:, 2] <= (tod+delta_tod))
    
    datatod = data_loc[masktod]
    
    if np.all(np.isnan(datatod[:, 10]))==True:
        print('lacking tod data, expanding search')
        delta = 1
        mask = (data[:, :, 0] >= long - delta) & (data[:, :, 0] <= long+delta) & (data[:, :, 1] >= lat - delta) & (data[:, :, 1] <= lat+delta)
        data_loc = data[mask, :]
        
        if tod < delta_tod:
            masktod = (data_loc[:, 2] > (tod - delta_tod)%24 ) | (data_loc[:, 2] < (tod+delta_tod)%24)
        elif tod > 24-delta_tod:
            masktod = (data_loc[:, 2] > (tod - delta_tod)%24 ) | (data_loc[:, 2] < (tod+delta_tod)%24)
        else:            
            masktod = (data_loc[:, 2] >= (tod - delta_tod)) & (data_loc[:, 2] <= (tod+delta_tod))
        
        datatod = data_loc[masktod]
        #raise Exception ('Lacking tod data')
    
    T = datatod[:, 10]
    
    if np.all(np.isnan(T)) == True:
        print('lacking tod data, expanding search')
        print(lat, long, tod)
        
        delta = 1.5
        delta_tod = 1

        mask = (data[:, :, 0] >= long - delta) & (data[:, :, 0] <= long+delta) & (data[:, :, 1] >= lat - delta) & (data[:, :, 1] <= lat+delta)
        data_loc = data[mask, :]
        
        if tod < delta_tod:
            masktod = (data_loc[:, 2] > (tod - delta_tod)%24 ) | (data_loc[:, 2] < (tod+delta_tod)%24)
        elif tod > 24-delta_tod:
            masktod = (data_loc[:, 2] > (tod - delta_tod)%24 ) | (data_loc[:, 2] < (tod+delta_tod)%24)
        else:            
            masktod = (data_loc[:, 2] >= (tod - delta_tod)) & (data_loc[:, 2] <= (tod+delta_tod))
        
        datatod = data_loc[masktod]
        #raise Exception ('Lacking tod data')
        #raise Exception("all T are NaN")
        
        T = datatod[:, 10]
        n = np.count_nonzero(~np.isnan(T))
        Tmean = np.nanmean(T)
    else:
        n = np.count_nonzero(~np.isnan(T))
        Tmean = np.nanmean(T)
    
    return Tmean, n

#%% Mechanisms
def random_vals():
    direction = random.uniform(0, 2*np.pi)
    launch = random.uniform(0, np.pi/2)
    return direction, launch

def loss(sigma_Sputtering, photo_lifespan, f_tof, tod):
    # Uses probability functions given time of a particle (in flight or sitting) with the lifetime from the literature
    # Considers sputtering and photodestruction. 
    
    # These mechanisms are sourced from solar UV/photons, so in theory they only occur on the lit portion
    if tod > 6 and tod < 18:
        cond1 = sputtering_prob(sigma_Sputtering, f_tof)
        cond2 = photodestruction_inflight_prob(photo_lifespan, f_tof)
        cond = np.any((cond1, cond2))
    else:
        # Loss only occurs during day time (an approximation for lit)
        cond = False
    return cond

def surftime(R_bar, Temperature, pMass):
    # desorption
    Ed = 0.6
    tau_surf = desorption_R21(Temperature, Ed)
    #tau_surf = residence_surface_T(2*10**12, 0.6, Temperature)

    Pv = clapyeron(triple_P, triple_T, R_bar, Lc, Temperature)
    sub = sublimation_surf(Pv, Temperature, pMass*Avo)

    perc_mono = 0.01
    theta_mon = 10**19 * perc_mono
    tau_sub = (theta_mon * pMass) / sub
    
    return np.minimum(tau_surf, tau_sub)

#%%
local_noon = 0
def Model_MonteCarlo(particle, dt, t, local_noon):
    R_bar = R/(m_H2O/1000)
    pMass = mass
    sigma_Sputtering = sput_lifespan_G19
    photo_lifespan =  photo_lifespan_S14

    # results array
    results = np.zeros((8, int(t/dt)+1))*np.nan # (lat, long, tod, temp, exists, total time, hops, dist), 2rd is time
    
    # if exists = True, set 1. Else, set 0
    
    i = 0
    results[0:3, 0] = particle
    results[4, 0] = False
    conda = False
    condb = False
    condc = False
    
    lunar_dt = sec_per_hour_M*dt

    # let particle run for 1 lunar rotation
    for i in range(0, int(t/dt), 1):     
        #print('Lunar time: %2.1f'%i)
        # find initial temperature from location
        results[3, i], n = DivinerT(results[0, i], results[1, i], results[2, i], data)
        #print('Surface temperature: %2.0f K '%results[3, i])
        
        # define how long it sits for
        tau_surf = surftime(R_bar, results[3, i], pMass)  

        # if it sits for a timestep, test lost, then move to next timestep
        if tau_surf >= lunar_dt:
            #print('Particle sits longer than a lunar time step')
            #print('Latitude: %2.0f, Longitude: %2.0f' %(np.rad2deg(results[0, i]), np.rad2deg(results[1, i])))
            conda = loss(sigma_Sputtering, photo_lifespan, lunar_dt, results[2, i])
            if conda == True:
                #print('Particle is lost from sitting')
                results[4, i] = conda
                break
            else:
                #print('Particle is not lost from sitting')
                results[0:3, i+1] = results[0:3, i]
                results[4, i+1] = conda
                results[5, i] = 0
                results[6, i] = 0
                results[7, i] = 0

        else: 
            #print('Particle begins jump')
            tof_tot = 0
            tot_time = 0
            tot_dist = 0
            #conda == False
            hops = 0
            while tot_time < lunar_dt:
                #print(tot_time, lunar_dt)
                # while the time of jumping is less than a timestep
                conda = loss(sigma_Sputtering, photo_lifespan, tau_surf, results[2, i])
                if conda == True:
                    #print('Particle is lost during sitting, following a jump')
                    # if lost to loss mechanism, exit loop. 
                    results[4, i+1] = conda
                    break
                else:
                    # let it bounce
                    direction, launch = random_vals() 
                    results[0:3, i+1], f_tof, distm, condc = ballistic(results[3, i], results[0, i], results[1, i], results[2, i], direction, launch, pMass, vel_dist)
                    #print(results[0:3, i+1], f_tof, distm, condc)
                    if condc == True:
                        #print('Particle is lost to jeans loss')
                        results[4, i+1] = condc
                        break
                    else:
                        condb = loss(sigma_Sputtering, photo_lifespan, f_tof, results[2, i])

                        hops += 1
                        tof_tot += f_tof
                        tot_dist += distm

                        # is it detroyed in the jump?
                        if condb == True:
                            #print('Particle is lost in jump')
                            results[4, i+1] = condb
                            break
                        else:
                            results[4, i+1] = condb
                            results[3, i+1], n = DivinerT(results[0, i+1], results[1, i+1], results[2, i+1], data)
                            tau_surf = surftime(R_bar, results[3, i+1], pMass)

                            # advance total time. 
                            tot_time += (f_tof + tau_surf)
                            
            results[5, i] = tof_tot
            results[6, i] = hops
            results[7, i] = tot_dist                
            #print('Particle ends jump after %3.2e'%tot_time)
            #print('Particle is at: (%2.1f, %2.1f), %2.1f hr'%(np.rad2deg(results[0, i]), np.rad2deg(results[1, i]), results[2, i]))
        
        if conda == True or condb == True or condc == True:
            #print('Particle is lost from the simulation')
            conda = False
            condb = False
            condc = False
            break
        else:
            #print('Moon rotates')
            local_noon += (360/(t/dt)) # degrees
            #print('Local noon: %2.1f'%local_noon)
            results[2, i+1] = (12 + (np.rad2deg(results[1, i+1])+local_noon)*(24/360))%24 
            #print('New time of day: %2.1f'%results[2, i+1])
    return results[:, :-1]

 

#%% Using Diviner Bolometric temperatures - this takes ~ 3 min if file doesn't exist
def Diviner_nan(data):
    mask = np.argwhere(data[:, 10] == -9999)
    data[mask, 10] = np.nan
    return data

loc = '/Users/laferrierek/Box Sync/Desktop/Research/Moon_Transport/Codes/Data/'
path_to_file = loc+'global_cuml_avg_cyl_90S_90N.npy'
path = Path(path_to_file)

if path.is_file():
    data = np.load(path_to_file, allow_pickle=True)

else:

    data0_10N = np.loadtxt(loc+'global_cumul_avg_cyl_00n10n_002.txt', delimiter=',', skiprows=1)
    data10_20N = np.loadtxt(loc+'global_cumul_avg_cyl_10n20n_002.txt', delimiter=',', skiprows=1)
    data20_30N = np.loadtxt(loc+'global_cumul_avg_cyl_20n30n_002.txt', delimiter=',', skiprows=1)
    data30_40N = np.loadtxt(loc+'global_cumul_avg_cyl_30n40n_002.txt', delimiter=',', skiprows=1)
    data40_50N = np.loadtxt(loc+'global_cumul_avg_cyl_40n50n_002.txt', delimiter=',', skiprows=1)
    data50_60N = np.loadtxt(loc+'global_cumul_avg_cyl_50n60n_002.txt', delimiter=',', skiprows=1)
    data60_70N = np.loadtxt(loc+'global_cumul_avg_cyl_60n70n_002.txt', delimiter=',', skiprows=1)
    data70_80N = np.loadtxt(loc+'global_cumul_avg_cyl_70n80n_002.txt', delimiter=',', skiprows=1)
    data80_90N = np.loadtxt(loc+'global_cumul_avg_cyl_80n90n_002.txt', delimiter=',', skiprows=1)
    
    data0_10S = np.loadtxt(loc+'global_cumul_avg_cyl_10s00s_002.txt', delimiter=',', skiprows=1)
    data10_20S = np.loadtxt(loc+'global_cumul_avg_cyl_20s10s_002.txt', delimiter=',', skiprows=1)
    data20_30S = np.loadtxt(loc+'global_cumul_avg_cyl_30s20s_002.txt', delimiter=',', skiprows=1)
    data30_40S = np.loadtxt(loc+'global_cumul_avg_cyl_40s30s_002.txt', delimiter=',', skiprows=1)
    data40_50S = np.loadtxt(loc+'global_cumul_avg_cyl_50s40s_002.txt', delimiter=',', skiprows=1)
    data50_60S = np.loadtxt(loc+'global_cumul_avg_cyl_60s50s_002.txt', delimiter=',', skiprows=1)
    data60_70S = np.loadtxt(loc+'global_cumul_avg_cyl_70s60s_002.txt', delimiter=',', skiprows=1)
    data70_80S = np.loadtxt(loc+'global_cumul_avg_cyl_80s70s_002.txt', delimiter=',', skiprows=1)
    data80_90S = np.loadtxt(loc+'global_cumul_avg_cyl_90s80s_002.txt', delimiter=',', skiprows=1)

    data0_10N = Diviner_nan(data0_10N)
    data10_20N = Diviner_nan(data10_20N)
    data20_30N = Diviner_nan(data20_30N)
    data30_40N = Diviner_nan(data30_40N)
    data40_50N = Diviner_nan(data40_50N)
    data50_60N = Diviner_nan(data50_60N)
    data60_70N = Diviner_nan(data60_70N)
    data70_80N = Diviner_nan(data70_80N)
    data80_90N = Diviner_nan(data80_90N)
    
    data0_10S = Diviner_nan(data0_10S)
    data10_20S = Diviner_nan(data10_20S)
    data20_30S = Diviner_nan(data20_30S)
    data30_40S = Diviner_nan(data30_40S)
    data40_50S = Diviner_nan(data40_50S)
    data50_60S = Diviner_nan(data50_60S)
    data60_70S = Diviner_nan(data60_70S)
    data70_80S = Diviner_nan(data70_80S)
    data80_90S = Diviner_nan(data80_90S)
    
    data = np.stack([data80_90S, data70_80S, data60_70S, data50_60S, data40_50S, data30_40S, data20_30S, data10_20S, data0_10S, data0_10N, data10_20N, data20_30N, data30_40N, data40_50N, data50_60N, data60_70N, data70_80N, data80_90N])
    
    # save it
    np.save(loc+'global_cuml_avg_cyl_90S_90N.npy', data, allow_pickle=True)
