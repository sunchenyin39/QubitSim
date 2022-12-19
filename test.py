import numpy as np
from matplotlib import pyplot as plt
import model
import function
import constant as ct

circuit = model.Circuit()
Amplitude = 0.00365
f01_Q1 = 4.7035E9
Envolope = 1-np.cos(2*np.pi*circuit.t_list/(circuit.t_end-circuit.t_start))
circuit.signal_1 = Amplitude*Envolope*np.cos(2*np.pi*f01_Q1*circuit.t_list)

# print(np.sqrt(8*circuit.E_j1*circuit.E_c1)/ct.H/1e9)

# print(circuit.M_Ec/ct.H/1e9)
# H_0=circuit.time_evolution_operator_calculation(1)
# eigenvalue,featurevector= np.linalg.eig(H_0)
# eigenvalue=np.real(eigenvalue)
# for i in range(27):
#     print(np.max(np.abs(featurevector[:,i])))
# print(np.sort((eigenvalue-min(eigenvalue))/ct.H/1E9))

# print(circuit.M_Ej_generator(0,0,0)/ct.H/1E9)

circuit.run()
print(circuit.time_evolution_operator_dressed_sub)

# eigenvalue,featurevector=circuit.transformational_matrix_generator()
# eigenvalue=np.real(eigenvalue)
# print(np.sort((eigenvalue-min(eigenvalue))/ct.H/1E9))
# print(np.real((featurevector)))
