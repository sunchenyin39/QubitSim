import numpy as np
import constant as ct

phi=0.39
R1=3000
R2=2000
alpha=min(R1,R2)/max(R1,R2)
e=1.6e-19
h=6.62e-34
phi0=h/(2*e)
Ij=2.8e-4/min(R1,R2)
Lj=phi0/2/np.pi/Ij/2/np.abs((1+alpha)/2*np.cos(np.pi*phi)+1j*(1-alpha)/2*np.sin(np.pi*phi))

M_L=np.array([[1.0583093202472239e-08,1,1],[1,3.641526041658358e-09,1],[1,1,1.1382405732020422e-08]])
M_Ej_0=6.62e-34/(4*np.pi**2*4*1.6e-19**2)/1e9/M_L
M_Ej=np.zeros([3,3])

for i in range(3):
    for j in range(3):
        if (i==j):
            M_Ej[i][j]=np.sum(M_Ej_0[:,i])
        else:
            M_Ej[i][j]=-M_Ej_0[i][j]

print(M_Ej)
