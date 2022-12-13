import numpy as np
from matplotlib import pyplot as plt
import model

circuit = model.Circuit()
Amplitude = 0.00365
f01_Q1 = 4.7035E9
Envolope = 1-np.cos(2*np.pi*circuit.t_list/(circuit.t_end-circuit.t_start))
circuit.signal_1 = Amplitude*Envolope*np.cos(2*np.pi*f01_Q1*circuit.t_list)

print(circuit.time_evolution_operator_calculation(1))

# plt.figure()
# plt.plot(circuit.t_list,circuit.signal_1)
# plt.show()
