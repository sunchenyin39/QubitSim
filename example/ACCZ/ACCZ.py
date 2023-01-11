import numpy as np
import QubitSim.model
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
circuit.phi_r1 = 0.0*np.pi
circuit.phi_r2 = 0.08*np.pi
circuit.phi_r3 = 0.33*np.pi

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
circuit.t_end = 100E-9
circuit.t_piece = 1E-11
circuit.operator_order_num = 4
circuit.trigonometric_function_expand_order_num = 8
circuit.exponent_function_expand_order_num = 15
circuit.picture_filename = "ACCZ.png"
circuit.npy_filename = "ACCZ.npy"
circuit.mode = 1
circuit.initial()
# ====================================================================
# 2.Setting signals.
N = len(circuit.t_list)
pulse_time = 100.0
rise_time = 0.0
amplitude = 0.0727293384688140
frequency = 0.129946838635065
wave_para = [0, 10000, 0, 0, 0, -1174.41312058193, 0, 0]
envolope = np.zeros(N)
waveform_ts = np.zeros(N)
wave_para_1 = wave_para[0]
for i in range(N):
    t_now = circuit.t_list[i]*1E9
    if (t_now < rise_time) and (t_now > 0):
        envolope[i] = (1-np.cos(t_now/rise_time*np.pi))/2*amplitude
        waveform_ts[i] = envolope[i]*np.cos(t_now*2*np.pi*frequency)
    elif (t_now >= rise_time) and (t_now <= rise_time+pulse_time):
        envolope[i] = 0
        for k in range(int(np.floor(len(wave_para)/2))):
            envolope[i] = envolope[i] + wave_para[k*2] * \
                (1-np.cos((k+1)*np.pi*(t_now-rise_time)/pulse_time))
            envolope[i] = envolope[i] + wave_para[k*2+1] * \
                (np.sin((k+1)*np.pi*(t_now-rise_time)/pulse_time))
        envolope[i] = (envolope[i]+1)*amplitude
        waveform_ts[i] = envolope[i]*np.cos(t_now*2*np.pi*frequency)
    elif (t_now > rise_time+pulse_time) and (t_now < 2*rise_time+pulse_time):
        envolope[i] = (1-np.cos((t_now-pulse_time)/rise_time *
                       np.pi))/2*amplitude*(1+2*wave_para_1)
        waveform_ts[i] = envolope[i]*np.cos(t_now*2*np.pi*frequency)
amp_norm = np.mean(envolope[1:])/(pulse_time+rise_time) * \
    (pulse_time+2*rise_time)/amplitude
envolope = envolope/amp_norm
waveform_ts = waveform_ts/amp_norm
circuit.signal_3z = waveform_ts
plt.figure()
plt.plot(circuit.t_list*1E9, circuit.signal_3z)
plt.title("signal_3z")
plt.xlabel("t/ns")
plt.tight_layout()
plt.savefig(fname="ACCZ_signal.png")
# 3.Run.
circuit.run()
# 4.data process
ACCZ_matrix = np.load(circuit.npy_filename)
print("ACCZ_matrix:")
for i in range(4):
    for j in range(4):
        print("%.4f" % np.abs(ACCZ_matrix[i][j]), end='_')
        print("%.4f" % np.angle(ACCZ_matrix[i][j]), end=',')
    print()
phase_globle = np.angle(ACCZ_matrix[0][0])
phase1 = np.angle(ACCZ_matrix[1][1])
phase2 = np.angle(ACCZ_matrix[2][2])
phase_U = np.diag([np.exp(-complex(0, 1)*phase_globle), np.exp(-complex(0, 1)*phase1),
                  np.exp(-complex(0, 1)*phase2), np.exp(-complex(0, 1)*(phase1+phase2-phase_globle))])
ACCZ_matrix_free_phase=np.matmul(phase_U,ACCZ_matrix)
print("\nACCZ_matrix(phase free):")
for i in range(4):
    for j in range(4):
        print("%.4f" % np.abs(ACCZ_matrix_free_phase[i][j]), end='_')
        print("%.4f" % np.angle(ACCZ_matrix_free_phase[i][j]), end=',')
    print()