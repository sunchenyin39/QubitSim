import numpy as np
import QubitSim.model

circuit = QubitSim.model.Circuit()

Amplitude = 0.00365
f01_Q1 = 4.7035E9
Envolope = 1-np.cos(2*np.pi*circuit.t_list/(circuit.t_end-circuit.t_start))
circuit.signal_1 = Amplitude*Envolope*np.cos(2*np.pi*f01_Q1*circuit.t_list)
circuit.picture_filename="idleQ1_X2PQ2.png"
circuit.npy_filename="idleQ1_X2PQ2.npy"
circuit.run()
