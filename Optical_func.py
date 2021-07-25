# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 00:44:50 2021

@author: SvrenA
"""
import math
import numpy as np
import scipy.constants as sp
from numpy.lib.scimath import sqrt as csqrt

class Optical_func:
    
    def __init__(self,Nx,Lx,wavelength,beamwaist,z,xt):
        self.Nx=Nx
        self.Lx = Lx
        self.wavelength = wavelength
        self.k = (2*sp.pi)/wavelength
        self.beamwaist = beamwaist
        self.z = z
        self.xt = xt
        self.deltax = self.Lx/self.Nx;
    
    def Sampling(self):  # sampling space
     x = np.arange(-self.Lx/2,self.Lx/2,self.deltax); y=x;
     X, Y = np.meshgrid(x,y);
     r = np.sqrt(np.square(X)+np.square(Y));
     rt = np.sqrt(np.square(X-self.xt)+np.square(Y));
     delfx = 1/self.Lx; 
     fx = np.arange(-1/(2*self.deltax),(1/(2*self.deltax)),delfx); fy=fx;
     FX,FY = np.meshgrid(fx,fy);
     rf = np.sqrt(np.square(FX)+np.square(FY));
     return(r,rt,FX,FY)
 
    def gaussian(self,r,rt):
     Zr = (sp.pi*self.beamwaist**2)/self.wavelength                            # rayleigh distance
     w = self.beamwaist*math.sqrt(1+np.square(self.z/Zr));                     # Beamradii
     R = self.z +(np.square(Zr)/self.z)                                        # radius of curvature
     phi = math.atan(self.z/Zr);                                               # Gouy phase
     # Farfield_div = math.degrees(wavelength/(sp.pi*Beamwaist));               # Farfield divergence
     gau1 = np.multiply(np.exp(-(np.square(r)/np.square(w))),(np.exp(-1j*self.k*self.z+1j*phi))); # Paraxial gaussian function
     gau1t = np.multiply(np.exp(-(np.square(rt)/np.square(w))),(np.exp(-1j*self.k*self.z+1j*phi))); # Tilted gaussian function
     gau = np.multiply(gau1,np.exp(-1j*self.k*(np.square(r)/(2*R))))
     gaut =  np.multiply(gau1t,np.exp(-1j*self.k*(np.square(rt)/(2*R))))
     return(gau,gaut)
 
    def ASM_func(self,FX,FY,Z):                                                # transfer function for beam propagation
       return(np.exp(-1j*self.k*(np.real(csqrt(1-(np.square(FX/self.k)+np.square(FY/self.k)))))*Z))
 
    def obj(self,w,physize):                                                   # object generation_chopper wheel
        nx = int(physize//self.deltax);                       
        p = int((self.Nx-nx)//2);                         
        if (nx % 2 == 0):                        
             p = p;
        else:
            nx = (physize//self.deltax)-1;
            p = (self.Nx-nx)//2;   
        li = np.linspace(-1,1,nx);
        Xi,Yi = np.meshgrid(li,li)
        rad = np.sqrt(Xi**2+Yi**2)<0.5
        th = np.arctan2(Yi,Xi)
        S= 1+np.sin(w*th)
        S = np.pad(S<1*rad,p,'constant')
        return(S)
        