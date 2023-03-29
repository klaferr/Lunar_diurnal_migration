#!/usr/bin/env python
# coding: utf-8

# In[10]:


#import all libraries
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sp
from scipy.stats import maxwell
import math
from scipy.stats import norm
import random
import scipy.optimize
import collections

#Setting Input Values
μ = (7.3065*(10**(-23)))/1000 #kg/molecule
r = 1738.1 * 1000 #m, radius of moon
M = 7.3477*(10**22) #kg, mass of moon
g_0 = 1.623 #m/s^2
F_SW = 3 * (10 ** 12) #1/m^2 * 1/s
h_imp = 10 ** -7 #m
Y_O = 0.1 #no units
C_O = 2.3 * (10 ** 28) # 1/m^3
P_T = 8.3 * (10 ** 4) #H2O photolysis time in s
mu = 0.018 #kg/mol, water
T_t = 273.16 #Kelvin, triple temp point
P_t = 611.730 #Paschals, triple pressure point
Q = 51058  #J/molecule, enthalpy
R = 8.3145 #J/(mol * K), gas constant

#Reading data for temperature given a lat,long, and time of day
path_to_file = 'global_cuml_avg_cyl_90S_90N.npy'
DI_data = np.load(path_to_file, allow_pickle=True)


"""def DivinerT(lat_R, long_R, tod, delta, data):

    lat = np.rad2deg(lat_R)
    long = np.rad2deg(long_R - np.pi)
    
    masklong = (data[:, :, 0] > long - delta) & (data[:, :, 0] < long+delta)
    masklat = (data[:, :, 1] > lat - delta) & (data[:, :, 1] < lat+delta)
    masktod = (data[:, :,  2] > tod - delta) & (data[:, :, 2] < tod+delta)

    row = masklong*masklat*masktod

    if np.any(row) != True:
        raise Exception("Lacking lat/long/tod data")

    else:
        T = data[row, 10]
        Tmean = np.nanmean(T)
    
        return Tmean, lat_R, long_R, tod

Temp, lat_R, long_R, tod = DivinerT(lat_R = 0, long_R = np.pi, tod = 12, delta = 0.75, data = DI_data)
"""

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
    
    # handle crossing 24 hr line
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
        print('lacking tod data, expanding search for second time')
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


#Creating Probability Distributions for velocity and finding most probable velocity
def probability_distribution(Temp,μ):  
    v= np.arange(0,905,1)


    s5 = 4*np.pi*(μ/(2*np.pi*sp.k*Temp))**(1.5)*(v**2)
    s6 = np.e** (((-1)*μ*v**(2))/(2*sp.k*Temp))
    s_mean_T = s5*s6
    
    mean_peak = random.choices(v, s_mean_T) #random velocity weighted choice
    #mean_peak = np.argmax(s_mean_T)
    mean_peak = mean_peak[0]
    
    return mean_peak

#mean_peak = probability_distribution(Temp,μ)


#Creating distance based off random angles of phi and most probable velocities
def ballistic_transport(mean_peak, r, M, phi):
    #phi = random.uniform(0,90)
    
    phi_0 = np.deg2rad(phi)
    
    a_mean = 1/((2/r)-((mean_peak**2)/(sp.G*M)))
    E_mean = (1-((r*(2*a_mean-r)*(np.sin(phi_0)**2))/(a_mean**2)))**(1/2)
    omega_mean = np.arccos((a_mean-a_mean*(E_mean**2)-r)/(r*E_mean))
    distance_mean = ((2*np.pi)-(2*omega_mean))*r

    
    return distance_mean

def time_of_flight(mean_peak, r, g_0):
    theta_array = np.linspace(0,90,1000)
    theta = random.choices(theta_array, weights = np.cos(theta_array))[0] 
    v_0 = mean_peak * np.sin(math.radians(theta))
    h_max = (r * (v_0 ** 2))/(2*r*g_0 - (v_0**2))
    a = ((v_0)**2)*r
    b = ((v_0)**2) - 2 * r * g_0
    u_1 = a + (b * h_max)
    u_2 = a
    v_1 = r + h_max
    v_2 = r
    l = a - (b * r)
    p_1 = ((2 * b * h_max) + a + (b * r))/ l
    p_2 = (a + (b * r))/ l
    
    
    if u_1 * v_1 < 0:
        while u_1 * v_1 < 0:
            theta = random.uniform(1,90)
            v_0 = mean_peak * math.radians(theta)
            h_max = (r * (v_0 ** 2))/(2*r*g_0 - (v_0**2))
            a = ((v_0)**2)*r
            b = ((v_0)**2) - 2 * r * g_0
            u_1 = a + (b * h_max)
            u_2 = a
            v_1 = r + h_max
            v_2 = r
            l = a - (b * r)
            p_1 = ((2 * b * h_max) + a + (b * r))/ l
            p_2 = (a + (b * r))/ l
        
    t_h_max = (2 * (v_1 / math.sqrt(v_1**2)) * ((math.sqrt(u_1*v_1)/b) + ((l/(2*b)) 
        * ((1/ (math.sqrt(-b))) * np.arcsin(p_1)))))

    t_0 = (2 * (v_2 / math.sqrt(v_2**2)) * ((math.sqrt(u_2*v_2)/b) + ((l/(2*b)) 
        * ((1/ (math.sqrt(-b))) * np.arcsin(p_2)))))

    t_flight = t_h_max - t_0
    return t_flight, theta

#t_flight, theta = time_of_flight(mean_peak, r, g_0)

def Photodestruction(t_flight, P_T):
    P_Destruction = 1 - (math.e ** (-t_flight/P_T))
    return P_Destruction

#P_Destruction = Photodestruction(t_flight, P_T)

def Sublimation(Temp, mu, T_t, P_t, Q, R):
    P_v = (P_t)* math.e ** ((-Q/(R) * ((1/Temp)- (1/T_t))))
    E = (P_v * (( mu)**(1/2)))/((2*math.pi * R * Temp)**(1/2)) #kg/m^2/s
    E_Gyr = (E  * 3.16 * (10 ** 16)) #kg/m^2/Gyr

    return E_Gyr, E

#E_Gyr, E = Sublimation(Temp, mu, T_t, P_t, Q, R)

def Sputtering(F_SW,h_imp,Y_O,C_O):
    Sputtering = (F_SW/h_imp) * (Y_O / C_O) # 1/s
    return Sputtering

#Sput = Sputtering(F_SW,h_imp,Y_O,C_O)


def Lat_Lon(in_lat, in_long, distance, radius, t_flight, tod):
    #given a distance and angle, calculate...
    # from Menten+ 2022
    # output in radians, input in radians
    travel_dir = random.uniform(0, 2*np.pi)
    lat2 = np.arcsin(np.sin(in_lat)*np.cos(distance/radius)+np.cos(in_lat)*np.sin(distance/radius)*np.cos(travel_dir))
    long2 = in_long + np.arctan2(np.sin(travel_dir)*np.sin(distance/radius)*np.cos(in_lat), np.cos(distance/radius)-np.sin(in_lat)*np.sin(lat2))
    #check new time calculation
    new_tod = tod + (t_flight/(3600 * 29.53)) + ((24/ (2 * np.pi)) * (long2 - in_long))
    return lat2, long2%(2*np.pi), new_tod %24


# In[31]:


def particle_jumping(i,p, particles):
    #setting total time
    results = np.zeros(8)
    total_t = 0
    jump_count = 0
    results[5] = particles[i,5,p-1]
    
        
    if np.isnan(particles[i,2,p-1]):
        results[6] = particles[i,6,p-1]
        #Put results[6] update here
        total_t = 10**6
        lat_R = -9999
        long_R = -9999
        tod = np.nan
    
    else: 
        
        #Caclulating coordinates, temp, and sublimation on ground
        if p ==0:
            Temp, n = DivinerT(particles[i,0,p] , particles[i,1,p], particles[i,2,p],  data = DI_data)
            results[7] = Temp
            lat_R = particles[i,0,p]
            long_R = particles[i,1,p]
            
            print("Lat: ", particles[i,0,p])
            print("Long: ", particles[i,1,p])
        else:
            Temp, n = DivinerT(particles[i,0,p-1] , particles[i,1,p-1], particles[i,2,p],  data = DI_data)
            results[7] = Temp
            lat_R = particles[i,0,p-1]
            long_R = particles[i,1,p-1]
            
            print("Lat: ", particles[i,0,p-1])
            print("Long: ", particles[i,1,p-1])

        print("Temp: ", Temp)
        print("tod: ", particles[i,2,p])
        tod = particles[i,2,p]
    
    
    
    
    while ((total_t) < ((10 ** 6)/4)):  #This line must be changed for time step length
        
        #print("while loop")

        #Caclulating coordinates, temp, and sublimation on ground
        E_Gyr, E = Sublimation(Temp, mu, T_t, P_t, Q, R)
        
        #print(E_Gyr)
        #print(Temp)
        
        #checks if sublimation occurs at new spot
        if (E_Gyr > 1):
            
            #print("sublimation")
            
            #Caclulating velocity, time of flight, and distance travelled with hop
            mean_peak = probability_distribution(Temp,μ)
            t_flight, theta = time_of_flight(mean_peak, r, g_0)
            distance_mean = ballistic_transport(mean_peak, r, M, theta)
            
            results[5] = results[5] + distance_mean
            #print(distance_mean)
            
            #Calculating chance of Photodestruction and Sputtering during hop
            P_Destruction = Photodestruction(t_flight, P_T)
            Sput = Sputtering(F_SW,h_imp,Y_O,C_O) * t_flight

            #Coming up with random "chance" to determine if particle is destroyed
            P_Destruction_Chance = random.uniform(0,1)
            Sput_Chance = random.uniform(0,1)

            #Checking if particle is destroyed
            if (P_Destruction >= P_Destruction_Chance) or (Sput >= Sput_Chance):
                if P_Destruction >= P_Destruction_Chance:
                    results[6] = 1
                if Sput >= Sput_Chance:
                    results[6] = 2
                #make tod nan, then break, preserve lat and long
                lat_R, long_R, tod = Lat_Lon(lat_R,long_R,distance_mean, r, t_flight, tod)
                tod = np.nan
                print("Particle destroyed, Particle Jumps: ", particles[i,3, p-1] + jump_count)
                break
    
            lat_R, long_R, tod = Lat_Lon(lat_R,long_R,distance_mean, r, t_flight, tod)
            Temp, n = DivinerT(lat_R , long_R, tod, data = DI_data)
            results[7] = Temp
            total_t = total_t + t_flight
            jump_count += 1
            
        else:
            #GO to next hour
            total_t = 10 ** 6
    
    
    results[0] = lat_R
    results[1] = long_R
    results[2] = tod
    results[3] = particles[i,3,p-1] + jump_count
    
    if np.isnan(tod):
        results[4] = particles[i,4,p-1]
        results[7] = np.nan
        print(results[4])
    else:
        results[4] = particles[i,4,p-1] + 1
        print(results[4])
    print("distance jumped: ", results[5])       
    return(results)
    


# In[40]:


#add times steps and number of time steps
#switch for loops so that it sends one particle through whole day

num_particles = 5
params = 8 #6
time_steps = 3

particles = np.zeros([num_particles,params,time_steps+1])




for i in range (0,num_particles):
    print("Particle ", i)
    #randomizes particles locations, tod = 0 at long = 0
    particles[i,0,0] = random.uniform(-np.pi/2, np.pi/2)      
    particles[i,1,0] = random.uniform(0, 2* np.pi) 
    particles[i,2,0] = ((particles[i,1,0])/(2 *np.pi)) * 24
    Temp, n = DivinerT(particles[i,0,0] , particles[i,1,0], particles[i,2,0],  data = DI_data)
    particles[i,7,0] = Temp
    
    print(particles[i,0,0])
    print(particles[i,1,0])
    print(particles[i,2,0])
    print(particles[i,4,0])
    for p in range (1,time_steps + 1, 1):
        print("Time step ", p)
        print("tod: ", particles[i, 2, p])
        particles[i, :, p] = particle_jumping(i,p, particles)
        
        if p < time_steps - 1:
            if np.isnan(particles[i, 2, p]):
                particles[i,2, p + 1] = np.nan
            else:
                particles[i,2, p + 1] = ((particles[i, 2, p] + 1) % 24) #tod update
            
            #print("Old tod ",  particles[i, 2, p])
            #print("New tod: ",  particles[i,2, p + 1])


# In[48]:


plt.hist(particles[:,3,2], 8)
plt.xlabel("Particles hops")
plt.ylabel("Number of particles")
plt.show()

counter = collections.Counter(particles[:,4,2])
plt.bar(list(counter.keys()),list(counter.values()))
plt.xlabel("Time Steps Survived")
plt.ylabel("Number of particles")
plt.show()

plt.hist(particles[:,5,2], 8)
plt.xlabel("Distance Travelled")
plt.ylabel("Number of particles")
plt.show()

z = 2 #particle reference
plt.scatter(180 / np.pi * particles[z,1,:], 180/np.pi * particles[z,0,:],c = particles[z,7,:], cmap = 'Spectral_r' ,vmin = 50, vmax = 400)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.xlim(0,360)
plt.ylim(-90,90)
plt.title("One particle (particle 1) position every time step")
plt.colorbar()
plt.show()

k = 0 #time step reference
plt.scatter(180/np.pi * particles[:,1,k], 180/np.pi * particles[:,0,k], c = particles[:,7,k], cmap = 'Spectral_r', vmin = 50, vmax = 400)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.xlim(0,360)
plt.ylim(-90,90)
plt.title("All particle positions at timestep: 0")
plt.colorbar()
plt.show()


# In[20]:


print(particles)
#Each column is a parameter; Columns: 0 = lat, 1 = long, 2 = local tod, 3 = jump count, 4 = universal tod, 5 = distance jumped, 6 = type of destruction, 7 = temp
#Each row is a new time step
#Each matrix is a new particle
#Last colum, 0 is still alive, 1 is photodestruction, 2 is sputtering


# In[ ]:


np.save("Particles Data", particles)


# In[ ]:



#saving the data
#np.save to save file
#if v > gravity then Jeans escape and add as type pf destruction
#tod in lat long map, replace temperature for colors for local TOD
#temp vs tod
#distance vs tod (distance formula from starting point to ending point, SPHERICAL (use lat,long))


#1000 particles for 96 timesteps (24 hours)
#

#Starting and ending lat/long
#make lat long map circular


# In[27]:





# In[ ]:


#np.argwhere(particles[:,0,0]< -1)

