import math
import numpy as np
from numpy.lib.scimath import sqrt as csqrt
import scipy.constants as sp
from scipy.fftpack import fft2, ifft2 , ifftshift, fftshift
from sympy import fwht,ifwht
import matplotlib.pyplot as plt
import time

start = time.time()
def plot2(variable,title,xmin,xmax) :
    plt.figure()
    plt.imshow(variable)  
    plt.xlim(xmin,xmax)
    plt.ylim(xmin,xmax)
    plt.title(title)
    plt.colorbar()
    plt.show()   
    
def Sampling(Lx,Nx):
    deltax = Lx/Nx;# sampling space
    x = np.arange(-Lx/2,Lx/2,deltax); y=x;
    X, Y = np.meshgrid(x,y);
    r = np.sqrt(np.square(X)+np.square(Y));
    delfx = 1/Lx; 
    fx = np.arange(-1/(2*deltax),(1/(2*deltax)),delfx); fy=fx;
    FX, FY = np.meshgrid(fx,fy);
    # rf = np.sqrt(np.square(FX)+np.square(FY));
    return(r,deltax,X,Y,fx, fy,FX,FY)
    
    
def gaussian(Beamwaist,z,wavelength,k,r,rt):
    Zr = (sp.pi*Beamwaist**2)/wavelength # rayleigh distance
    w = Beamwaist*math.sqrt(1+np.square(z/Zr)); # Beamradii
    R = z + (np.square(Zr)/z) # radius of curvature
    phi = math.atan(z/Zr); # Gouy phase
    # Farfield_div = math.degrees(wavelength/(sp.pi*Beamwaist)); # Farfield divergence
    gau1 = np.multiply(np.exp(-(np.square(r)/np.square(w))),(np.exp(-1j*k*z+1j*phi))); # Paraxial gaussian function
    gau1t = np.multiply(np.exp(-(np.square(rt)/np.square(w))),(np.exp(-1j*k*z+1j*phi))); # Tilted gaussian function
    return(np.multiply(gau1,np.exp(-1j*k*(np.square(r)/(2*R)))),np.multiply(gau1t,np.exp(-1j*k*(np.square(rt)/(2*R)))))


def object(w,physize,Nx,deltax):
    nx = int(physize//deltax);                       
    p = int((Nx-nx)//2);                         
    if (nx % 2 == 0):                        
         p = p;
    else:
        nx = (physize//deltax)-1;
        p = (Nx-nx)//2;   
    li = np.linspace(-1,1,nx);
    Xi,Yi = np.meshgrid(li,li)
    rad = np.sqrt(Xi**2+Yi**2)<0.5
    th = np.arctan2(Yi,Xi)
    S= 1+np.sin(w*th)
    S = np.pad(S<1*rad,p,'constant')
    return(S)
    
    
def ASM_func(wavelen,k,Fx,Fy,Z):
   return(np.exp(1j*k*(np.real(csqrt(1-(np.square(Fx*wavelen)+np.square(Fy*wavelen)))))*Z))
    

               
frequency = 0.35e12; # in THz
wavelength=sp.nu2lambda(frequency)*1; # wavelength in meters
k = (2*sp.pi)/wavelength; # wavenumber in rad/meters

### Sampling =============================================================================
Lx = 50e-3; # Window size
Nx = 320; # Pixels
r,deltax,X,Y,fx,fy,FX,FY = Sampling(Lx,Nx)
# plot2(r,'radial',fx,fy)
# =============================================================================

## Gaussian =============================================================================
Beamwaist = 0.75e-3; # beamwaist in meters
z = 10e-3 #propagation distance in meters
xt = -15e-3
rt = np.sqrt(np.square(X-xt)+np.square(Y));
gau,gaut = gaussian(Beamwaist,z,wavelength,k,r,rt)
# plot2(abs(gau),'propagated Gaussian beam z=10 mm',0,Nx)

# =======================================================================================

## Object =============================================================================
physize = 40e-3;
w = 5; # Number of spokes
obj=object(w,physize,Nx,deltax) # Object 
# plot2((obj),'Object',0,Nx) 
# =============================================================================


## Beam Propagation to the SPI Unit including Holography =============================================================================
angle = 25 # tilt angle
prop = np.zeros(Nx)
tilt = np.exp(1j*k*math.sin(math.radians(angle))*X); # tilt function
prop = ifft2((fftshift(fft2(gaut*tilt))*ASM_func(wavelength,k,FX,FY,25e-3))) # reference beam propagation to the mask plane
U = ifft2(fftshift(fft2(gau*obj))*ASM_func(wavelength,k,FX,FY,20e-3)) # object beam propagation to the mask plane
hologram= (prop + U)
# plot2(abs(U),'Diffracted Object beam',0,Nx)
# plot2(abs(prop),'Reference beam',0,Nx)
plot2(abs(hologram),'Hologram',0,Nx)
# =============================================================================

### Mask & SPI Measurements =============================================================================
blk_size = 32;
fact =10; # block size
Measurements = blk_size*blk_size;
delta = np.zeros(Measurements); t = np.zeros(Measurements);h = np.zeros(Measurements)
p = int((Nx-(blk_size*fact))/2)
for i in range(0,Measurements-1):
    delta[i]=1
    maskp = (np.reshape(0.5*(np.ones(Measurements)+fwht(delta)),(blk_size,blk_size))).astype(np.float64);
    m_mask = np.kron(maskp,np.ones((fact,fact)));
    m_mask = np.pad(m_mask,p,'constant')
    t[i] = np.sum(abs(np.multiply((hologram),m_mask)))  # object seen by one set of masks
    h[i] = np.sum(abs(np.multiply((hologram),(1-m_mask))))  # object seen by the complimentary set
    delta = np.zeros(Measurements);
# =============================================================================

## Reconstruction =============================================================================
rec = (ifwht(t-h)) # applying ifwht on obtained 1d data
for i in range(0,Measurements):
    rec[i]=float(rec[i])
res = np.reshape(rec,(blk_size,blk_size))
plot2((res),'SPI Reconstructed Image',0,blk_size) # SPI Reconstructed image
# =============================================================================

## Holography reconstruction =============================================================================
r,deltax,X,Y,fx,fy,FX,FY = Sampling(Lx,blk_size)
tiltr =  np.exp(-1j*k*math.sin(math.radians(angle))*X); # tilt correction
holo_four = fftshift(fft2(abs(res)*tiltr)) # SPI reconstructed object in fourier domain
#Filtering
a =380.30
mas = np.zeros(Nx)
mas=np.sqrt(np.square(FX)+np.square(FY))<=a # circular binary filter
Back_prop=(ifft2((mas*holo_four)*ASM_func(wavelength,k,FX,FY,20e-3))) # reconstruction from backpropagation
Back_propimag = np.arctan2(np.imag(Back_prop),np.real(Back_prop)) # Phase profile
# plot2(abs(mas),'Circular Binary radius',0,blk_size)
# plot2(abs(holo_four),'Reconstructed hologram in Fourier domain',0,blk_size)
plot2(abs(Back_prop),'Reconstructed object from backpropagation',0,blk_size)

# =============================================================================
    
end = time.time()
print('Execution time in secs:', (end-start))   
    
