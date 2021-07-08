
import math
import numpy as np
from numpy.lib.scimath import sqrt as csqrt
from numpy.linalg import norm
import scipy.constants as sp
from scipy.fftpack import fft2, ifft2 , ifftshift, fftshift
from scipy.linalg import hadamard
import matplotlib.pyplot as plt
import time

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
   return(np.exp(-1j*k*(np.real(csqrt(1-(np.square(Fx/k)+np.square(Fy/k)))))*Z))

def fwht(x,st):
    N= len(x)
    if (N&N-1 == 0):
       n = math.log2(N)
       if st =='hadamard':
           x=ordering(x,2)
       else:
           x = x 
       for i in range(0,N-1,2):
           x[i] = x[i] + x[i+1];
           x[i+1] = x[i] - 2 * x[i+1];
    L=1
    y1=np.zeros(x.shape,dtype=float)
    stage=2
    while(stage<=int(n)):
        M = pow(2,L)
        K = 0;J=0 
        if (st == 'sequency'):
            while(K<N):
                for j in range(J,J+M-1,2):
                    y1[K]   = x[j]   +  x[j+M];
                    y1[K+1] = x[j]   -  x[j+M];
                    y1[K+2] = x[j+1] -  x[j+1+M];
                    y1[K+3] = x[j+1] +  x[j+1+M];
                    K = K + 4;
                J = J + 2*M
        else:
             while(K<N):
                for j in range(J,J+M-1,2):
                    y1[K]   = x[j]   +  x[j+M];
                    y1[K+1] = x[j]   -  x[j+M];
                    y1[K+2] = x[j+1] +  x[j+1+M];
                    y1[K+3] = x[j+1] -  x[j+1+M];
                    K = K + 4;
                J = J + 2*M    
        x= y1.copy()
        L+=1;
        stage+=1
    y1=y1/N   
    return(y1)

def ifwht(y,st):
    y1=fwht(y,st)
    y1 = y1*y1.shape[0]
    return(y1)


def dectobase(num,base): #converting decimal to binary
    length1=len(num)
    num=np.linspace(0,length1-1,length1)
    length=num.size
    n=int(np.max(math.log2(max(num+1))))
    s=np.zeros([length,n])
    s[:,n-1]=(num%2)
    nm=n-1
    while(np.any(num) and nm>0):
        nm-=1
        num=num//2
        s[:,nm]=(num%2)
    s=(np.array(s,dtype=int))
    # s=np.fliplr(np.array(s,dtype=int))
    schar=["".join(item) for item in s.astype(str)]
    return(schar)   

def basetodec(s,base):
    num_str1 = s#num_str[::-1]
    length = len(num_str1)
    d=np.zeros([length,1])
    num = 0
    for i in range(0,length):
        num=0
        num_str=num_str1[i]
        for k in range(len(num_str)):
            dig = num_str[k]
            if dig.isdigit():
                dig = int(dig)
            else:    #Assuming its either number or alphabet only
                dig = ord(dig.upper())-ord('A')+10
            num += dig*(base**k)
        d[i]=num
        d= (np.array(d,dtype=int))
    return(d)

def ordering(num,base):
    x = basetodec(dectobase(num,base),base)
    sq = np.zeros(num.shape)
    i=0
    while(i<num.shape[0]):
        d = x[i]
        sq[i]=num[d]
        i+=1
    return(sq)
    

def threshold(x,theta):
    return(np.sign(x)*(np.maximum(0,np.abs(x)-theta)))

         
frequency = 0.35e12; # in THz
wavelength=sp.nu2lambda(frequency)*1; # wavelength in meters
k = (2*sp.pi)/wavelength; # wavenumber in rad/meters

### Sampling =============================================================================
Lx = 50e-3; # Window size
Nx = 128; # Pixels
r,deltax,X,Y,fx,fy,FX,FY = Sampling(Lx,Nx)
# =============================================================================

## Gaussian =============================================================================
Beamwaist = 0.75e-3; # beamwaist in meters
z = 10e-3 #propagation distance in meters
xt = -15e-3
rt = np.sqrt(np.square(X-xt)+np.square(Y));
gau,gaut = gaussian(Beamwaist,z,wavelength,k,r,rt)
plot2(abs(gau),'propagated Gaussian beam z=10 mm',0,Nx,0)
# =======================================================================================

## Object =============================================================================
physize = 30e-3;
w = 5; # Number of spokes
obj=object(w,physize,Nx,deltax) # Object 
plot2((obj),'Object',0,Nx,0) 
# =============================================================================


## Beam Propagation to the SPI Unit including Holography =============================================================================
x = ifft2(fftshift(fft2(obj))*ASM_func(wavelength,k,FX,FY,20e-3)) # object beam propagation to the mask plane
plot2(abs(x),'Diffracted Object beam',0,Nx,1)
# =============================================================================

### Mask & SPI Measurements =============================================================================
blk_size = 128;
fact =1; # block size 
Measurements = 80*80#blk_size*blk_size
del1=np.zeros([16384,16384]);
t = np.zeros([blk_size*blk_size,1]);h = np.zeros([blk_size*blk_size,1]);y_meas = np.zeros([blk_size*blk_size,1]);
delta = np.matrix(hadamard(blk_size*blk_size));
delta = ifwht(fwht(delta,'hadamard'),'sequency') # conversion from natural to sequency ordering
del1=delta;
i=0;
while i<Measurements-1:
    mas = np.kron((np.reshape(0.5*(1+delta[i]),(blk_size,blk_size))),np.ones([fact,fact]))
    t[i]=np.sum(np.multiply(mas,abs(x)));
    h[i]=np.sum(np.multiply(1-mas,abs(x)));
    y_meas[i] = t[i]-h[i]
    i+=1
plt.plot(y_meas)
del1[i:(blk_size*blk_size)]=0;
# # =============================================================================

# ## Reconstruction =============================================================================
rec_x=ifwht(np.array(y_meas),'sequency') # calculates inverse fwht
rec=np.reshape((rec_x),(blk_size,blk_size))
plot2(abs(rec),'SPI Reconstructed Image',0,blk_size,1) # SPI Reconstructed image
# # =============================================================================

# Implementing ISTA Code=============================================================================
# Initialization
# Minimization of LASSO

k_iter = 50;
Lambda = 1e-16; # 1e-5
L = 5e-2; # 5e-2
x_rec = np.zeros(y_meas.shape); min_x = np.zeros(k_iter);
for i in range(0,k_iter):
    x_rec = threshold(np.array(x_rec)-(1/L)*(np.dot(del1.T,(np.dot(del1,np.array(x_rec))-np.array(y_meas)))),Lambda/L)
    min_x[i] = (1/2)*np.square(norm((np.dot(del1,np.array(x_rec))-np.array(y_meas)),2))+Lambda*norm(np.array(x_rec),1)
p = np.reshape(np.array(x_rec),(blk_size,blk_size))
plot2(abs(p),'Reconstructed using ISTA 20 iter',0,blk_size,1) # SPI Reconstructed image
plt.plot(min_x) 
plt.xlabel("Number of Iterations")
plt.ylabel("Objective Function")

# =============================================================================
# Normalization

d= abs(x)-np.min(abs(x)) # input object 
dnorm= d/np.max(d);
pd= abs(p)-np.min(abs(p)) # reconstructed image using ISTA
pnorm= pd/np.max(pd);
pr=abs(rec)-np.min(abs(rec)) # Linear reconstructed image normalized
rnorm= pr/np.max(pr);
norm(rnorm-dnorm,2)# min error between the reconstructed image and actual object


# =============================================================================
end = time.time()
print('Execution time in secs:', (end-start)) 
# =============================================================================

   


