import numpy as np
from matplotlib import pyplot as plt
import model
import function
import constant as ct

circuit = model.Circuit()
# Amplitude = 0.00365
# f01_Q1 = 4.7035E9
# Envolope = 1-np.cos(2*np.pi*circuit.t_list/(circuit.t_end-circuit.t_start))
# circuit.signal_1 = Amplitude*Envolope*np.cos(2*np.pi*f01_Q1*circuit.t_list)

H_0=circuit.Hamiltonian_calculation(0)
eigenvalue,featurevector= np.linalg.eig(H_0)
for i in range(27):
    print(np.max(np.abs(featurevector[:,i])))

# eigenvalue=eigenvalue+1E-20
# print((eigenvalue-min(eigenvalue))/ct.H/1E9)

# print(np.sqrt(8*circuit.E_j1*circuit.E_c1*np.cos(circuit.phi_r1+np.pi*circuit.M_z_1*circuit.signal_1[0]/ct.PHI_ZERO))/ct.H*2*np.pi/1E9)
# print(np.sqrt(8*circuit.E_j2*circuit.E_c2*np.cos(circuit.phi_r2+np.pi*circuit.M_z_2*circuit.signal_2[0]/ct.PHI_ZERO))/ct.H*2*np.pi/1E9)
# print(np.sqrt(8*circuit.E_j3*circuit.E_c3*np.cos(circuit.phi_r3+np.pi*circuit.M_z_3*circuit.signal_3[0]/ct.PHI_ZERO))/ct.H*2*np.pi/1E9)