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
import time
#%% Define values for inputs
# Constants
Avo = 6.0221*10**(23)               # 1/ mol
G = 6.6743*10**(-11)                # m3/kg/s2
kb = 1.38064*10**(-23)              # J/K
R = 8.3145                          # J/mol/K
vel_dist = np.arange(0, 3000, 1)

# constants
sigma = 5.67*10**(-8) #W/m2/K4
A = 0.1 # mare, visual. ~0.3 for Highlands, visual
So = 1367 # solar flux, W/m2
epsilon = 0.95

# Molecules
m_H2O= 18.01528                     # g/ mol
m_OH = 17.008                       # g/mol
mass_H2O = (m_H2O/1000) / Avo           # kg/mol / (1/mol)
mass_OH = (m_OH/1000) / Avo
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

## Residence timescalex

# Photodestruction
photo_rate_S14 = 1.26*10**(-5)                      # 1/s.
photo_lifespan_S14 = 1/photo_rate_S14               # s

## OH
photo_rate_S23_OH = 7.49*10**(-6) # Smolka et al. 2023, 1/s
## H2O
photo_rate_S14_H2O = 1.26*10**(-5) # 1/s

# Sputtering
sput_rate_G19 = 1.3*10**(-10)           # 1/s
sput_lifespan_G19 = 1/sput_rate_G19
## OH
sput_rate_G19_OH = 1.3*10**(-10) # 1/S
## H2O
sput_rate_S23_H2O = 2.4*10**(-8) #1/s, Smolka 2023

# h2o on h2o, Prem et al. 2015
deltaH = 6.6*10**(-20) # J, binding energy for H2O on H2O matrix
v0 = 2*10**(12) #1/s, lattice vibrational frequency of H2O within H2O matrix)

# other people have opinions on the prem values

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
    
    hmax = (moonR*voy**2)/(2*moonR*g-voy**2)
    # if hmax > 6.15*10**7: #equation: a*(moonM/3*earthM)**(1/3); where a = 384400 km, 
        # raise Exception("exceed's hill sphere!")
        # print("hmax:", hmax)
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

# Photodissociation
def photodestruction_inflight_prob(tau, tof, cosi):
    # this is equiv. to 1-e^(-t/tau), where if happens < prob, it is lost
    # here, it is if happens > prob, it is lost
    
    #probability = (1-cosi)*np.exp(-tof/(tau)) #/cosi))
    probability = np.exp(-tof/(tau)) 
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

def cosi_smooth(lat, delta, hr):
    cosi = np.sin(lat)*np.sin(delta) + np.cos(lat)*np.cos(delta)*np.cos(hr)
    return cosi

#%% Temperature from data
def DivinerT(lat_r, long_r, tod, data):
    delta = 0.26*2
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
    launch_range = np.linspace(0, np.pi/2, 1000)
    launch = random.choices(launch_range, weights=np.cos(launch_range))[0]
    return direction, launch

def loss(sigma_Sputtering, photo_lifespan, f_tof, tod, cosi):
    # Uses probability functions given time of a particle (in flight or sitting) with the lifetime from the literature
    # Considers sputtering and photodestruction. 
    
    # These mechanisms are sourced from solar UV/photons, so in theory they only occur on the lit portion
    if tod >= 6 and tod <= 18:
        cond1 = sputtering_prob(sigma_Sputtering, f_tof)
        cond2 = photodestruction_inflight_prob(photo_lifespan, f_tof, cosi)

        #cond2 = photodestruction_inflight_prob(photo_lifespan, f_tof)
        cond = np.any((cond1, cond2))
    else:
        # Loss only occurs during day time (an approximation for lit)
        cond = False
    return cond

def surftime(R_bar, Temperature, pMass):
    # desorption
    Ed = 0.6
    tau_surf = desorption_R21(Temperature, Ed)

    Pv = clapyeron(triple_P, triple_T, R_bar, Lc, Temperature)
    sub = sublimation_surf(Pv, Temperature, pMass*Avo)

    perc_mono = 0.01
    theta_mon = 10**19 * perc_mono
    tau_sub = (theta_mon/Avo*(m_H2O/1000)) / sub
    
    return np.minimum(tau_surf, tau_sub)

#%% Smooth model
def Model_MonteCarlo(particle, dt, t, local_noon, molecule):
    if molecule == "H2O":
        R_bar = R/(m_H2O/1000)
        pMass = mass_H2O
        sigma_Sputtering = 1/sput_rate_S23_H2O
        photo_lifespan =  1/photo_rate_S14_H2O
    elif molecule == "OH":
        R_bar = R/(m_H2O/1000)
        pMass = mass_H2O
        sigma_Sputtering = 1/sput_rate_G19_OH
        photo_lifespan =  1/photo_rate_S23_OH
        
    else:
        raise Exception("Must be OH or H2O")
    
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
        # st_time = time.time()
        #print('Lunar time: %2.1f'%i)
        # find initial temperature from location
        results[3, i], n = DivinerT(results[0, i], results[1, i], results[2, i], data)
        #print('Surface temperature: %2.0f K '%results[3, i])
        
        # define how long it sits for
        tau_surf = surftime(R_bar, results[3, i], pMass)  
        
        # define incidence angle
        cosi = cosi_smooth(results[0, i], 0, results[2, i])

        # if it sits for a timestep, test lost, then move to next timestep
        if tau_surf >= lunar_dt:
            #print('Particle sits longer than a lunar time step')
            #print('Latitude: %2.0f, Longitude: %2.0f' %(np.rad2deg(results[0, i]), np.rad2deg(results[1, i])))
            conda = loss(sigma_Sputtering, photo_lifespan, lunar_dt, results[2, i], cosi)
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
                conda = loss(sigma_Sputtering, photo_lifespan, tau_surf, results[2, i], cosi)
                if conda == True:
                    #print('Particle is lost during sitting, following a jump')
                    # if lost to loss mechanism, exit loop. 
                    results[4, i+1] = conda
                    break
                else:
                    # let it bounce
                    direction, launch = random_vals() 
                    results[0:3, i+1], f_tof, distm, condc = ballistic(results[3, i], results[0, i], results[1, i], results[2, i], direction, launch, pMass, vel_dist)
                    cosi = cosi_smooth(results[0, i+1], 0, results[2, i+1])
                    #print(results[0:3, i+1], f_tof, distm, condc)
                    if condc == True:
                        #print('Particle is lost to jeans loss')
                        results[4, i+1] = condc
                        break
                    else:
                        condb = loss(sigma_Sputtering, photo_lifespan, f_tof, results[2, i], cosi)

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
        #en = time.time()
        #print('Time of step %3.0f is: %3.2f'%(i, en-st_time))
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

 
#%% Rubanenko et al. (2020) roughness surface temperatures   

# Functions
def beta(S0, A, r):
    # S0: solar constant at 1 AU
    # A: Surface albedo
    # r: distance from the sun in AU
    return S0*(1-A)/(r)**2

def rho(sigma, epsilon, beta):
    return sigma*epsilon/beta

def solar_zenith(lat, delta, h):
    # phi: local latitude
    # delta: declination of the sun
    # h: hour angle (local solar time) (from noon)
    cos_z = np.sin(lat)*np.sin(delta) + np.cos(lat)*np.cos(delta)*np.cos(h)
    return np.arccos(cos_z)

def solar_azimuth(lat, d, hr, z):
    # d is declination
    # h is 
    # z is solar zenith
    cos_az = (np.sin(d)*np.cos(lat)-np.cos(hr)*np.cos(d)*np.sin(lat))/(np.sin(z))
    return np.arccos(cos_az)

def cosi_rough(lat, delta, h, theta, alpha):
    # a is solar azimuth angle
    # theta is slope aspect
    # alpha is slope angle
    # delta is declination, ==0
    
    z = solar_zenith(lat, delta, h)
    a = solar_azimuth(lat, delta, h, z)
    cos_z = np.cos(z)
    cosi = np.cos(alpha)*cos_z + np.sin(alpha)*np.sin(z)*np.cos(theta-a)    
    return cosi

def f_cosi_theta(omega, i, z):
    c_p = np.cos(i + z)
    c_n = np.cos(i - z)
    coeff1 = omega/(np.sqrt(np.pi*((1-(c_p/c_n)))))
    coeff2 = 1+ (1/(omega**2 * c_n**2))
    expo = (1/(2*omega**2))*(1-(1/(c_n**2)))
    return coeff1*coeff2*np.exp(expo)

def f_F(F, beta, z, omega):
    phi1 = np.sqrt(1-((F**2)/(beta**2)))
    phi2 = ((F/beta)*(1/np.tan(z))) / phi1
    phi3 = phi1**2 * (1+phi2)**2 * (np.sin(z))**2

    coeff1 = omega /(np.sqrt(2*np.pi*beta**2))
    coeff2 = np.sqrt(1+phi2*(1/np.tan(z)))/phi1
    coeff3 = (1+(1/(omega**2 * phi3)))
    expo = (1/(2*omega**2) *(1 - (1/phi3)))
    return coeff1*coeff2*coeff3*np.exp(expo)

def f_T(T, omega, rho, z):
    tau1 = np.sqrt(1-((rho**2) * (T**8)))
    tau2 = (rho*T**4)/(tau1 * np.tan(z))
    tau3 = tau1**2 * (1+tau2)**2 * (np.sin(z))**2

    coeff1 = (4*omega*rho*(T**3)) / (np.sqrt(2*np.pi))
    coeff2 = np.sqrt(1+tau2*(1/np.tan(z)))/tau1
    coeff3 = (1+(1/(omega**2 * tau3)))
    expo =  (1/(2*omega**2)) * (1-(1/tau3)) #(1- (1/(tau1**2 * tau3)))  

    return coeff1*coeff2*coeff3*np.exp(expo)


def LOLA(lat, long, data):
    # grab the RMS slope at that lat, long
    # w, h: 2880, 5760 
    # one row for each 0.0625 degrees of latitude
    
    res = 0.0625
    lat_d = np.rad2deg(lat)
    long_d = np.rad2deg(long)
    X = int((long_d-180)/res)
    Y = int((lat_d+90)/res)
    
    omega = data[X, Y]
    return omega
    # read in LOLA bidirectional surface roughness from map
    

def roughT(lat, long, tod, data, LOLAdata):

    if tod < 6 or tod > 18:
        #phi = np.arccos(np.cos((np.deg2rad((tod*15 - 180)))))
        # Prem et al. (2018)
        #a = np.array([444.738, -448.937, 239.668, -63.8844, 8.34064, -0.423502])
        #sums = np.zeros(5)
        #for j in range(0, 5):
        #    sums[j] = a[j]*phi**j
        #T = np.sum(sums) + 35*(np.sin(np.abs(lat))-1)
        temp_flat, n = DivinerT(lat, long, tod, data) 
        T = temp_flat
        w = 0
        #print('Nighttime:', temp_flat, T)
        
    else:
        # Rubanenko et al. (2021)
        wbi_LOLA = LOLA(lat, long, LOLAdata)
        w = np.tan(np.deg2rad(wbi_LOLA))/np.sqrt(2)
        
        #theta = np.deg2rad(0) # slope aspect
        #a = np.deg2rad(0) # the orientation of the sun 
        i_arr = np.deg2rad(np.linspace(1, 90, 180))
        
        # solar_zenith angle:
        #z = np.deg2rad(15) # solar zenith angle
        hr_angle = np.deg2rad((tod*15)-180)
        z = solar_zenith(lat, 0, hr_angle)
        if z == 0:
            print("zenith is zero, special case")
        
        # expected cosi       
        #fcosi_n = f_cosi_theta(w, i_arr, z)/np.max(f_cosi_theta(w, i_arr, z))
    
        # F:
        F_arr = (So*(1-A)/((rEarth)**2))*np.cos(i_arr)
        B = F_arr/np.cos(i_arr)
        #PDF_F = f_F(F_arr, B, z, w)
    
        # T:
        p = sigma*epsilon/B
        T_arr = (F_arr/(sigma*epsilon))**(1/4)
        PDF_T = f_T(T_arr, w, p, z)
        
        norm_weights = PDF_T/np.trapz(PDF_T)
        
        T = random.choices(T_arr, weights=norm_weights)[0]
        #print('Daytime:', temp_flat, T_arr[np.argwhere(PDF_T==np.nanmax(PDF_T))], T)
    
    return T, w


def Model_MonteCarlo_Rough(particle, dt, t, local_noon, molecule, omegaT):
    if molecule == "H2O":
        R_bar = R/(m_H2O/1000)
        pMass = mass_H2O
        sigma_Sputtering = 1/sput_rate_S23_H2O
        photo_lifespan =  1/photo_rate_S14_H2O
    elif molecule == "OH":
        R_bar = R/(m_H2O/1000)
        pMass = mass_H2O
        sigma_Sputtering = 1/sput_rate_G19_OH
        photo_lifespan =  1/photo_rate_S23_OH
        
    else:
        raise Exception("Must be OH or H2O")
    
    #roughness data
    loladata = omegaT

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
        #st_time = time.time()
        #print('Lunar time: %2.1f'%i)

        results[3, i], alpha = roughT(results[0, i], results[1, i], results[2, i], data, loladata)
        #print('Surface temperature: %2.0f K '%results[3, i])

        # define how long it sits for
        tau_surf = surftime(R_bar, results[3, i], pMass)  

        cosi = cosi_rough(results[0, i], 0, results[2, i], 1/(np.pi*2), alpha)
        
        # if it sits for a timestep, test lost, then move to next timestep
        if tau_surf >= lunar_dt:
            #print('Particle sits longer than a lunar time step')
            #print('Latitude: %2.0f, Longitude: %2.0f' %(np.rad2deg(results[0, i]), np.rad2deg(results[1, i])))
            conda = loss(sigma_Sputtering, photo_lifespan, lunar_dt, results[2, i], cosi)
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
                conda = loss(sigma_Sputtering, photo_lifespan, tau_surf, results[2, i], cosi)
                if conda == True:
                    #print('Particle is lost during sitting, following a jump')
                    # if lost to loss mechanism, exit loop. 
                    results[4, i+1] = conda
                    break
                else:
                    # let it bounce
                    direction, launch = random_vals() 
                    results[0:3, i+1], f_tof, distm, condc = ballistic(results[3, i], results[0, i], results[1, i], results[2, i], direction, launch, pMass, vel_dist)
                    
                    cosi = cosi_rough(results[0, i], 0, results[2, i], theta, alpha)

                    if condc == True:
                        #print('Particle is lost to jeans loss')
                        results[4, i+1] = condc
                        break
                    else:
                        condb = loss(sigma_Sputtering, photo_lifespan, f_tof, results[2, i], cosi)

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
                            results[3, i+1] = roughT(results[0, i+1], results[1, i+1], results[2, i+1], data, loladata)
                            tau_surf = surftime(R_bar, results[3, i+1], pMass)
                            #print(tau_surf, f_tof)
                            # advance total time. 
                            tot_time += (f_tof + tau_surf)
                            
            results[5, i] = tof_tot
            results[6, i] = hops
            results[7, i] = tot_dist                
            #print('Particle ends jump after %3.2e'%tot_time)
            #print('Particle is at: (%2.1f, %2.1f), %2.1f hr'%(np.rad2deg(results[0, i]), np.rad2deg(results[1, i]), results[2, i]))
        #en = time.time()
        #print('Time of step %3.0f is: %3.2f'%(i, en-st_time))
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

#%% With a production mechanism 
def production_cosi(n, local_noon):
    # how do we define production
    ## highest proton flux (and in theory, formed water) would be at subsolar point

    long_range = np.deg2rad(np.arange(0, 360, 1)) # this is in radians

    # find cosi
    long_start = int((local_noon - 90)%360) #int(np.rad2deg(np.min(long_range[cosi > 0])) %360)
    long_stop = int((local_noon + 90)%360) #int(np.rad2deg(np.max(long_range[cosi > 0])) %360)

    lat_range = np.deg2rad(np.arange(-90, 90))
    # create particles
    latweight = np.cos(lat_range)
    longweight = -np.cos(np.deg2rad(np.arange(90, 270)))

    # fails if local noon < 90 or greater than 270 degrees because of the limitations of the range function
    if local_noon < 90 :
        long_range = np.concatenate((range(long_start, 360), range(0, long_stop)))
    elif local_noon >= 270:
        long_range = np.concatenate((range(long_start, 360), range(0, long_stop)))
    else:
        long_range = range(long_start, long_stop)

    particles = np.zeros((n, 3)) # latitude, longitude,  tod
    particles[:, 0] = np.deg2rad(random.choices(range(-90, 90), weights=latweight, k=n)) # latitude in radians
    particles[:, 1] = np.deg2rad(random.choices(long_range, weights=longweight , k=n)) # longitude in radians
    particles[:, 2] = ((12+(((np.rad2deg(particles[:, 1])-local_noon))*24)/360))%24 # tod, based on where local noon is

    return particles

# Smooth model with production
def Model_MonteCarlo_produce(particle, dt, t, ms, on, local_noon, molecule):
    if molecule == "H2O":
        R_bar = R/(m_H2O/1000)
        pMass = mass_H2O
        sigma_Sputtering = 1/sput_rate_S23_H2O
        photo_lifespan =  1/photo_rate_S14_H2O
    elif molecule == "OH":
        R_bar = R/(m_H2O/1000)
        pMass = mass_H2O
        sigma_Sputtering = 1/sput_rate_G19_OH
        photo_lifespan =  1/photo_rate_S23_OH
        
    else:
        raise Exception("Must be OH or H2O")
    
    # number of particles
    n = on
    
    # Number of added particles
    ms = 10 # particles per step 
    m = int((t/dt)*ms) # TOTAL added particles
    
    # results array
    results = np.zeros((8, int(t/dt)+1))*np.nan # (lat, long, tod, temp, exists, total time, hops, dist), 2rd is time
    
    # if exists = True, set 1. Else, set 0
    results[0:3, 0] = particle
    results[4, 0] = False

    for i in range(0, int(t/dt)-1, 1):
        print('time step', i)
        conda = False
        condb = False
        condc = False
        
        lunar_dt = sec_per_hour_M*dt
        
        n = len(results[:, 0, i][~np.isnan(results[:, 0, i])])
        print('%2.0f particles to be run'%n)
        
        for nn in range(0, n-1, 1):
            
            if results[nn, 4, i] == 1 or np.isnan(results[nn, 4, i]):
                #print('Particle already lost, skip')
                pass
    
            else:
                #print('Particle number:', nn)
                # find initial temperature from location
                results[nn, 3, i], num = DivinerT(results[nn, 0, i], results[nn, 1, i], results[nn, 2, i], data)
                #print('Surface temperature: %2.0f K '%results[3, i])
    
                # define how long it sits for
                tau_surf = surftime(R_bar, results[nn, 3, i], pMass)  
    
                # define incidence angle
                cosi = cosi_smooth(results[nn, 0, i], 0, results[nn, 2, i])
    
                # if it sits for a timestep, test lost, then move to next timestep
                if tau_surf >= lunar_dt:
                    #print('Particle sits longer than a lunar time step')
                    #print('Latitude: %2.0f, Longitude: %2.0f' %(np.rad2deg(results[0, i]), np.rad2deg(results[1, i])))
                    conda = loss(sigma_Sputtering, photo_lifespan, lunar_dt, results[nn, 2, i], cosi)
                    if conda == True:
                        #print('Particle is lost from sitting')
                        results[nn, 4, i] = conda
                        results[nn, 4, i+1] = conda
    
                        #break
                    else:
                        #print('Particle is not lost from sitting')
                        results[nn, 0:3, i+1] = results[nn, 0:3, i]
                        results[nn, 4, i+1] = conda
                        results[nn, 5, i] = 0
                        results[nn, 6, i] = 0
                        results[nn, 7, i] = 0
    
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
                        conda = loss(sigma_Sputtering, photo_lifespan, tau_surf, results[nn, 2, i], cosi)
                        if conda == True:
                            #print('Particle is lost during sitting, following a jump')
                            # if lost to loss mechanism, exit loop. 
                            results[nn, 4, i+1] = conda
                            break
                        else:
                            # let it bounce
                            direction, launch = random_vals() 
                            results[nn, 0:3, i+1], f_tof, distm, condc = ballistic(results[nn, 3, i], results[nn, 0, i], results[nn, 1, i], results[nn, 2, i], direction, launch, pMass, vel_dist)
                            cosi = cosi_smooth(results[nn, 0, i+1], 0, results[nn, 2, i+1])
                            #print(results[0:3, i+1], f_tof, distm, condc)
                            if condc == True:
                                #print('Particle is lost to jeans loss')
                                results[nn, 4, i+1] = condc
                                break
                            else:
                                condb = loss(sigma_Sputtering, photo_lifespan, f_tof, results[nn, 2, i], cosi)
    
                                hops += 1
                                tof_tot += f_tof
                                tot_dist += distm
    
                                # is it detroyed in the jump?
                                if condb == True:
                                    #print('Particle is lost in jump')
                                    results[nn, 4, i+1] = condb
                                    break
                                else:
                                    results[nn, 4, i+1] = condb
                                    results[nn, 3, i+1], num = DivinerT(results[nn, 0, i+1], results[nn, 1, i+1], results[nn, 2, i+1], data)
                                    tau_surf = surftime(R_bar, results[nn, 3, i+1], pMass)
    
                                    # advance total time. 
                                    tot_time += (f_tof + tau_surf)
    
                    results[nn, 5, i] = tof_tot
                    results[nn, 6, i] = hops
                    results[nn, 7, i] = tot_dist                
                    #print('Particle ends jump after %3.2e'%tot_time)
                    #print('Particle is at: (%2.1f, %2.1f), %2.1f hr'%(np.rad2deg(results[0, i]), np.rad2deg(results[1, i]), results[2, i]))
                #en = time.time()
                #print('Time of step %3.0f is: %3.2f'%(i, en-st_time))
        
            if conda == True or condb == True or condc == True:
                print('Particle %4.0f is lost from the simulation'%nn)
                results[nn, 4, i] = True
                results[nn, 4, i+1] = True
    
                conda = False
                condb = False
                condc = False
    
        else:
            print('Moon rotates')
            local_noon += (360/(t/dt)) # degrees
            #print('Local noon: %2.1f'%local_noon)
            results[nn, 2, i+1] = (12 + (np.rad2deg(results[nn, 1, i+1])+local_noon)*(24/360))%24 
            #print('New time of day: %2.1f'%results[2, i+1])
        
            # now, add particles
            if n+ms*i > on+m:
                pass
            else:
                print("added %2.0f particles"%ms)
                results[n+ms*i:n+ms*(i+1), 0:3, i+1] = production_cosi(ms, local_noon)
                results[n+ms*i:n+ms*(i+1), 4, i+1] = np.zeros((ms))
            
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
    


    
