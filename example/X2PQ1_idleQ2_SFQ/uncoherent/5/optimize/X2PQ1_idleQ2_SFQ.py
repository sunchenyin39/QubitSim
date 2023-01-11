import numpy as np
import progressbar
import QubitSim.model
import QubitSim.SFQLab
from matplotlib import pyplot as plt

# 0.Build a quantum circuit.
circuit = QubitSim.model.Circuit()
# ====================================================================
# 1.Parameter list.
# Capacitor:
# C_1: The left qubit's capacitor.
# C_2: The right qubit's capacitor.
# C_3: The middle coupler's capacitor.
# C_12: The capacitor between left qubit and right qubit.
# C_23: The capacitor between right qubit and middle coupler.
# C_13: The capacitor between left qubit and middle coupler.
circuit.C_1 = 88.1E-15+1E-20
circuit.C_2 = 88.1E-15+1E-20
circuit.C_3 = 125.4E-15+1E-20
circuit.C_12 = 6E-16+1E-20
circuit.C_23 = 10.11E-15+1E-20
circuit.C_13 = 10.11E-15+1E-20

# L_off: Off inductance between low coupled notes.
circuit.L_off = 1

# Room temperature resistor of Josephson junction:
# V_test: Voltage using to test room temperature resister.
# R_1_1: The room temperature resistor of first junction of left qubit's DCSQUID.
# R_1_2: The room temperature resistor of second junction of left qubit's DCSQUID.
# R_2_1: The room temperature resistor of first junction of right qubits's DCSQUID.
# R_2_2: The room temperature resistor of second junction of right qubits's DCSQUID.
# R_3_1: The room temperature resistor of first junction of middle coupler's DCSQUID.
# R_3_2: The room temperature resistor of second junction of middle coupler's DCSQUID.
circuit.V_test = 2.8E-4
circuit.R_1_1 = 18000
circuit.R_1_2 = 18000
circuit.R_2_1 = 18000
circuit.R_2_2 = 18000
circuit.R_3_1 = 3000
circuit.R_3_2 = 2000

# Remanence phase:
# phi_r1: The remanence phase of left qubits's DCSQUID.
# phi_r2: The remanence phase of right qubit's DCSQUID.
# phi_r3: The remanence phase of middle coupler's DCSQUID.
# PS: phi_ri/2/pi*PHI_ZERO=Phi_ri.
circuit.phi_r1 = 0.0*np.pi
circuit.phi_r2 = 0.12*np.pi
circuit.phi_r3 = 0.39*np.pi

# Mutual inductance:
# M_z_1: The mutual inductance between signal line 1 and left qubit's DCSQUID.
# M_z_2: The mutual inductance between signal line 2 and right qubit's DCSQUID.
# M_z_3: The mutual inductance between signal line 3 and middle coupler's DCSQUID.
# M_x_1: The mutual inductance between signal line 1 and left qubit's main loop.
# M_x_2: The mutual inductance between signal line 2 and right qubit's main loop.
# M_x_3: The mutual inductance between signal line 3 and middle coupler's main loop.
circuit.M_z_1 = 2E-12
circuit.M_z_2 = 2E-12
circuit.M_z_3 = 2E-12
circuit.M_x_1 = 2E-12
circuit.M_x_2 = 2E-12
circuit.M_x_3 = 2E-12

# t_start: Starting time point.
# t_end: Ending time point.
# t_piece: Piece time.
# operator_order_num.
# trigonometric_function_expand_order_num.
# exponent_function_expand_order_num.
# picture_filename: Filename of picture to be drawed.
# npy_filename: Filename of subspace quantum gate.
circuit.t_start = 0
circuit.t_end = 40E-12
circuit.t_piece = 2E-12
circuit.operator_order_num = 4
circuit.trigonometric_function_expand_order_num = 8
circuit.exponent_function_expand_order_num = 15
circuit.picture_filename = "X2PQ1_idleQ2_SFQ.png"
circuit.npy_filename = "X2PQ1_idleQ2_SFQ.npy"
circuit.initial()
# ====================================================================
# 2.Getting transformational matrix converting bare bases to dressed bases.
# dressed_eigenvalue: Dressed states' energy eigenvalue.
# dressed_featurevector: Transformational matrix converting bare bases to dressed bases
dressed_eigenvalue, dressed_featurevector = circuit.transformational_matrix_generator()
E_00 = dressed_eigenvalue[0]
E_01 = dressed_eigenvalue[1]
E_10 = dressed_eigenvalue[2]
E_11 = E_10+E_01-E_00
np.save("X2PQ1_idleQ2_dressed_eigenvalue.npy", dressed_eigenvalue)
np.save("X2PQ1_idleQ2_dressed_featurevector.npy", dressed_featurevector)
# ====================================================================
# 3.Simulation calculating time evolution operator without signals.
read_mode = 0
if read_mode == 0:
    p = progressbar.ProgressBar()
    time_evolution_operator_0 = np.eye(circuit.operator_order_num**3)
    print("Calculating the whole time evolution operator without signals:")
    for i in p(range(int(circuit.t_piece_num/2))):
        time_evolution_operator_0 = np.matmul(
            circuit.time_evolution_operator_calculation(i+1), time_evolution_operator_0)
    np.save("X2PQ1_idleQ2_SFQ_0_matrix.npy", time_evolution_operator_0)
else:
    time_evolution_operator_0 = np.load("X2PQ1_idleQ2_SFQ_0_matrix.npy")
# ====================================================================
# 4.Setting signals for SFQ pulse
Amplitude = 0.03
t_center = 20E-12
t_width = 4E-12
circuit.signal_1 = QubitSim.SFQLab.Gaussian_function_list_generator(
    circuit.t_list, Amplitude, t_center, t_width)
plt.figure()
plt.plot(circuit.t_list*1E9, circuit.signal_1)
plt.title("signal_1")
plt.xlabel("t/ns")
plt.tight_layout()
plt.savefig(fname="X2PQ1_idleQ2_SFQ_signal.png")
# ====================================================================
# 5.Simulation calculating time evolution operator with SFQ pulse.
read_mode = 0
if read_mode == 0:
    p = progressbar.ProgressBar()
    time_evolution_operator_1 = np.eye(circuit.operator_order_num**3)
    print("Calculating the whole time evolution operator with SFQ pulse:")
    for i in p(range(int(circuit.t_piece_num/2))):
        time_evolution_operator_1 = np.matmul(
            circuit.time_evolution_operator_calculation(i+1), time_evolution_operator_1)
    np.save("X2PQ1_idleQ2_SFQ_1_matrix.npy", time_evolution_operator_1)
else:
    time_evolution_operator_1 = np.load("X2PQ1_idleQ2_SFQ_1_matrix.npy")
