# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 01:14:53 2021

@author: SvrenA
"""

import math
import numpy as np
from numpy.lib.scimath import sqrt as csqrt
from numpy.linalg import norm
import scipy.constants as sp
from scipy.fftpack import fft,ifft, fft2, ifft2 , ifftshift, fftshift
from scipy.linalg import hadamard
import matplotlib.pyplot as plt
import time
from my_fwht import fwht, ifwht
from Optical_func import Optical_func
from ISTA import ISTA

start = time.time()
def plot2(variable,title,xmin,xmax,flag) :
    if flag ==1:
        plt.figure()
        plt.imshow(variable)  
        plt.xlim(xmin,xmax)
        plt.ylim(xmin,xmax)
        plt.title(title)
        plt.colorbar()
        plt.show() 

# setup parameters_Initialized    
frequency = 0.35e12;                                                           # in THz
wavelength=sp.nu2lambda(frequency)*1;                                          # wavelength in meters
### Sampling parameters =============================================================================
Lx = 50e-3;                                                                    # Window size in meters
Nx =128                                                                        # sampling
## Gaussian beam parameters=============================================================================
Beamwaist = 0.75e-3;                                                           # beamwaist in meters
z = 50e-3                                                                      #propagation distance in meters
xt = -15e-3                                                                    # initial position for reference beam
g = Optical_func(Nx,Lx,wavelength,Beamwaist,z,xt)                              
r,rt,FX,FY = g.Sampling()
gau,gaut= g.gaussian(r,rt)  
plot2(abs(gau),'Diffracted Object beam',0,Nx,0)
## Object ===========================================================================================================
physize = 30e-3;
w = 5; # Number of spokes
obj= g.obj(w,physize) # Object 
plot2((obj),'Object',0,Nx,0) 
## Beam Propagation to the SPI Unit including Holography =============================================================================
x = ifft2(fftshift(fft2(obj*gau))*g.ASM_func(FX,FY,20e-3))                     # object beam propagation to the mask plane
plot2(abs(x),'Diffracted Object beam',0,Nx,1)
### Mask & SPI Measurements =============================================================================
blk_size =128;                                                                 # block size 
fact = 1; 
Measurements =96*96                                                            #Total no of measurements
mask ='hadamard'
dim=blk_size*blk_size;                                                         #Total no of pixels
#define random hadamard masks
num=np.linspace(0,dim-1,dim)
num2= np.random.permutation(np.array(num,dtype=int))
masks_ind = num2[0: Measurements]                                              #Mask indices choosen randomly        
if mask == 'hadamard':
    delta = hadamard(dim)#hdamard
    d=delta[masks_ind]#hadamard
elif mask == 'Bernoulli':
    delta  = np.random.randint(2,size=(Measurements,dim))
    dinv = np.linalg.pinv(np.dot(delta.T,delta))    
# the object
x_obj   = np.reshape(x,(Nx*Nx,1))                                              #Object reshaped to column vector
def A(x):
    if mask == 'hadamard':
        tmp=ifwht(x,'hadamard');                                                   #forward transform
        return(tmp[masks_ind])
    elif mask == 'Bernoulli':
        tmp = np.dot(delta,x)
        return tmp
    else:
        #tmp=ifft(x);
        return(tmp[masks_ind])

def At(y):
    if mask == 'hadamard':
        tmp = np.zeros([dim,1],dtype=complex);                                     #transpose of forward transform
        tmp[masks_ind]=y;
        return fwht(tmp,'hadamard');
    elif mask == 'Bernoulli':
        #tmp = np.dot(np.linalg.inv(delta),y)# inverse transform M=N^2
        tmp = np.dot(np.dot(dinv,delta.T),y)# pseudoinverse for M<N^2
        #tmp = np.dot(delta.T,y)#transpose matrix
        return tmp
    else:
        tmp = np.zeros([dim,1],dtype=complex);                                     #transpose of forward transform
        tmp[masks_ind]=y;
        return fft(tmp);
    
# (2) create the measurements
y_meas = A(x_obj);                                                             #measurement vector 
plt.plot(y_meas) 
# Implementing ISTA Code=============================================================================
k_iter = 1000;                                                                   # no of iterations
L = np.linalg.eigenvals(np.dot(d.T,d),ord=2)                                       #Lipschitz constant
#L=np.linalg.norm(d**2,ord=2)
Lambda = 0.1*max(np.abs(At(y_meas)))                                           # hyperparameter
x_rec,min_x = ISTA(k_iter,y_meas,A,At,L,Lambda,dim).forward()                  #ISTA 
p = np.reshape(np.array(x_rec),(blk_size,blk_size))
plot2(abs(p),'Reconstructed using ISTA '+str(k_iter) +' iterations' ,0,blk_size,1) 
plt.plot(min_x) 
plt.xlabel("Number of Iterations")
plt.ylabel("Objective Function")
# =============================================================================
end = time.time()
print('Execution time in secs:', (end-start)) 








