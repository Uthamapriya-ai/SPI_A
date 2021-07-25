# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 15:31:46 2021

@author: SvrenA
"""
import numpy as np
from numpy.linalg import norm

class ISTA:
       
    def __init__(self,k_iter,y,A,AT,L,Lambda,dim):
        self.k_iter =  k_iter
        self.y = y
        self. phi = A 
        self.phi_T = AT
        self. beta= 1/L
        self.Lambda = Lambda
        self.theta = Lambda/L
        self.dim = dim
        
        
    def forward(self):
        x= np.zeros([self.dim,1])
        min_x= np.zeros([self.k_iter,1])
        for i in range(self.k_iter):
             r = self.phi(np.array(x))-self.y;
             x= soft_threshold((np.array(x)-self.beta*self.phi_T(r)),self.theta)
             min_x[i]= (1/2)*np.square(norm(r,2))+self.Lambda*norm(x,1)
             return(x,min_x)
                     
def soft_threshold(xt,theta):
    e_real=np.sign(np.real(xt))*np.maximum(0,np.abs(np.real(xt))-theta)
    e_imag=np.sign(np.imag(xt))*np.maximum(0,np.abs(np.imag(xt))-theta)
    return(e_real+1j*e_imag)           
        