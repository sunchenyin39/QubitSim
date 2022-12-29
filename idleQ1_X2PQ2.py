import numpy as np
import QubitSim.model
import QubitSim.constant

circuit = QubitSim.model.Circuit()

Amplitude = 0.00365
f02_Q1 = 4.529869779999999E9
Envolope = 1-np.cos(2*np.pi*circuit.t_list/(circuit.t_end-circuit.t_start))
circuit.signal_2 = Amplitude*Envolope*np.cos(2*np.pi*f02_Q1*circuit.t_list)
circuit.picture_filename="idleQ1_X2PQ2.png"
circuit.npy_filename="idleQ1_X2PQ2.npy"
circuit.run()

# dressed_eigenvalue, dressed_featurevector=circuit.transformational_matrix_generator()
# print(dressed_eigenvalue/QubitSim.constant.H/1E9)
