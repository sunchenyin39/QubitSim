import numpy as np
import QubitSim.model
from scipy import special
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
circuit.C_1 = 4.8E-14+1E-20
circuit.C_2 = 4.8E-14+1E-20
circuit.C_3 = 9.6E-14+1E-20
circuit.C_12 = 3.5E-16+1E-20
circuit.C_23 = 8E-15+1E-20
circuit.C_13 = 8E-15+1E-20

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
circuit.R_1_1 = 22000
circuit.R_1_2 = 22000
circuit.R_2_1 = 22000
circuit.R_2_2 = 22000
circuit.R_3_1 = 4000
circuit.R_3_2 = 2000

# Remanence phase:
# phi_r1: The remanence phase of left qubits's DCSQUID.
# phi_r2: The remanence phase of right qubit's DCSQUID.
# phi_r3: The remanence phase of middle coupler's DCSQUID.
# PS: phi_ri/2/pi*PHI_ZERO=Phi_ri.
circuit.phi_r1 = 0.15*np.pi
circuit.phi_r2 = 0.175*np.pi
circuit.phi_r3 = 0.38*np.pi

# Mutual inductance:
# M_z_1: The mutual inductance between signal line 1 and left qubit's DCSQUID.
# M_z_2: The mutual inductance between signal line 2 and right qubit's DCSQUID.
# M_z_3: The mutual inductance between signal line 3 and middle coupler's DCSQUID.
# M_x_1: The mutual inductance between signal line 1 and left qubit's main loop.
# M_x_2: The mutual inductance between signal line 2 and right qubit's main loop.
# M_x_3: The mutual inductance between signal line 3 and middle coupler's main loop.
circuit.M_z_1 = 1E-12
circuit.M_z_2 = 1E-12
circuit.M_z_3 = 1E-12
circuit.M_x_1 = 1E-12
circuit.M_x_2 = 1E-12
circuit.M_x_3 = 1E-12

# t_start: Starting time point.
# t_end: Ending time point.
# t_piece: Piece time.
# operator_order_num.
# trigonometric_function_expand_order_num.
# exponent_function_expand_order_num.
# picture_filename: Filename of picture to be drawed.
# npy_filename: Filename of subspace quantum gate.
circuit.t_start = 0
circuit.t_end = 40E-9
circuit.t_piece = 1E-11
circuit.operator_order_num = 4
circuit.trigonometric_function_expand_order_num = 8
circuit.exponent_function_expand_order_num = 15
circuit.picture_filename = "DCCZ.png"
circuit.npy_filename = "DCCZ.npy"
circuit.initial()
# ====================================================================
# 2.Setting signals.


def flat_top(time_list, amplitude, rise_time_norm, rise_delta):
    rise_time = rise_time_norm*rise_delta
    T = time_list[-1]-2*rise_time
    waveform_list = (special.erf((time_list-rise_time)/rise_delta/np.sqrt(2)) -
                     special.erf((time_list-(T+rise_time))/rise_delta/np.sqrt(2)))
    if (np.max(waveform_list)-np.min(waveform_list)) > 1E-5:
        waveform_list = amplitude * \
            (waveform_list-np.min(waveform_list)) / \
            (np.max(waveform_list)-np.min(waveform_list))
    return waveform_list


amplitude_Q1 = 0
amplitude_Q2 = 0.0169
amplitude_C = 0.082
circuit.signal_1z = flat_top(circuit.t_list*1E9, amplitude_Q1, 1, 2)
circuit.signal_2z = flat_top(circuit.t_list*1E9, amplitude_Q2, 1, 2)
circuit.signal_3z = flat_top(circuit.t_list*1E9, amplitude_C, 1, 2)
plt.figure()
plt.subplot(2, 2, 1)
plt.plot(circuit.t_list*1E9, circuit.signal_1z)
plt.subplot(2, 2, 2)
plt.plot(circuit.t_list*1E9, circuit.signal_2z)
plt.subplot(2, 2, 3)
plt.plot(circuit.t_list*1E9, circuit.signal_3z)
plt.show()
# 3.Run.
circuit.run()
