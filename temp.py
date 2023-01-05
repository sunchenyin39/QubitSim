import numpy as np
from scipy import special

a = np.load("X2PQ1_idleQ2.npy")
print(a)
for i in range(4):
    for j in range(4):
        print("%.4f" % np.abs(a[i][j]), end='_')
        print("%.4f" % np.angle(a[i][j]), end=',')
    print()
    
b=np.zeros([2,2],dtype=complex)
b[0][0]=a[0][0]
b[0][1]=a[0][2]
b[1][0]=a[2][0]
b[1][1]=a[2][2]
print(b)

theta_g=(np.angle(b[0][0])+np.angle(b[1][1]))/2.0
phi=2*np.arccos(np.real(b[0][0]/np.exp(complex(0,1)*theta_g)))
nz=np.imag(b[0][0]/np.exp(complex(0,1)*theta_g))/(-1)/np.sin(phi/2)
nx=np.imag(b[0][1]/np.exp(complex(0,1)*theta_g))/(-1)/np.sin(phi/2)
ny=np.real(b[0][1]/np.exp(complex(0,1)*theta_g))/(-1)/np.sin(phi/2)
print(theta_g)
print(phi)
print(nx)
print(ny)
print(nz)
print(nx**2+ny**2+nz**2)

