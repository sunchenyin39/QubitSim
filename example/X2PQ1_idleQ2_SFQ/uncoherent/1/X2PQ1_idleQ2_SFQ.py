import numpy as np
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
circuit.t_end = 16E-9
circuit.t_piece = 2E-12
circuit.operator_order_num = 4
circuit.trigonometric_function_expand_order_num = 8
circuit.exponent_function_expand_order_num = 15
circuit.picture_filename = "X2PQ1_idleQ2_SFQ.png"
circuit.npy_filename = "X2PQ1_idleQ2_SFQ.npy"
circuit.initial()
# ====================================================================
# 2.Setting signals.
Amplitude = 0.03
# t_center = 0.04E-9*np.array([1, 2, 3, 6, 7, 8, 11, 12, 13, 17, 18, 22, 23, 24, 27, 28, 29, 33, 34, 38,
#                             39, 40, 43, 44, 45, 49, 50, 54, 55, 56, 59, 60, 61, 65, 66, 70, 71, 72, 75, 76, 77, 81, 82, 86, 87, 88, 91, 92, 93, 96, 97, 98, 102, 103, 104, 107, 108, 109, 112, 113, 114, 118, 119, 123, 124, 125, 128, 129, 130,
#                             134, 135,139,140,141,144,145,146,150,151,155,156,157,160,161,162,166,167,171,172,173,176,177,178,
#                             182,183,187,188,192,193,194,197,198,199,
#                             203,204,208,209,210,213,214,215,
#                             219,220,224,225,226,229,230,231,235,236,240,241,242,
#                             245,246,247,251,252,256,257,258,261,262,263,
#                             267,268,272,273,274,277,278,279,
#                             283,284,288,289,290,293,294,295,298,299,300,
#                             304,305,309,310,311,314,315,316,
#                             320,321,325,326,327,330,331,332,
#                             336,337,341,342,343,346,347,348,
#                             352,353,357,358,359,
#                             362,363,364,368,369,373,374,375,378,379,380])
t_center = 0.04E-9*np.array([1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 17, 18,19, 22, 23, 24,25,26, 27, 28, 29,30, 33, 34, 38,
                            39, 40, 43, 44, 45, 49, 50, 54, 55, 56, 59, 60, 61, 65, 66, 70, 71, 72, 75, 76, 77, 81, 82, 86, 87, 88, 91, 92, 93, 96, 97, 98, 102, 103, 104, 107, 108, 109, 112, 113, 114, 118, 119, 123, 124, 125, 128, 129, 130,
                            134, 135, 139, 140, 141, 144, 145, 146, 150, 151, 155, 156, 157, 160, 161, 162, 166, 167, 171, 172, 173, 176, 177, 178,
                            182, 183, 187, 188, 192, 193, 194, 197, 198, 199,
                            203, 204, 208, 209, 210, 211, 213, 214, 215,
                            219, 220, 224, 225, 226, 229, 230, 231, 235, 236, 240, 241, 242,
                            245, 246, 247, 251, 252, 256, 257, 258, 261, 262, 263,
                            267, 268, 272, 273, 274, 277, 278, 279,
                            283, 284, 288, 289, 290, 293, 294, 295, 298, 299, 300,
                            304, 305, 309, 310, 311, 314, 315, 316,
                            320, 321, 325, 326, 327, 330, 331, 332,
                            336, 337, 341, 342, 343, 346, 347, 348,
                            352, 353, 357, 358, 359,
                            362, 363, 364, 368, 369, 373, 374, 375, 378, 379, 380, 383, 384, 385, 389, 390,394,395,396])
t_width = 4E-12
circuit.signal_1 = QubitSim.SFQLab.Gaussian_function_sequence_generator(
    circuit.t_list, Amplitude, t_center, t_width)
# circuit.signal_1z = QubitSim.SFQLab.Gaussian_function_sequence_generator(
#     circuit.t_list, Amplitude, t_center, t_width)
plt.figure()
plt.plot(circuit.t_list*1E9, circuit.signal_1)
plt.title("signal_1")
plt.xlabel("t/ns")
plt.tight_layout()
plt.savefig(fname="X2PQ1_idleQ2_signal.png")
# 3.Run.
circuit.run()
# 4.Matrix display
X2PQ1_idleQ2_matrix = np.load(circuit.npy_filename)
print("X2PQ1_idleQ2_matrix:")
for i in range(4):
    for j in range(4):
        print("%.4f" % np.abs(X2PQ1_idleQ2_matrix[i][j]), end='_')
        print("%.4f" % np.angle(X2PQ1_idleQ2_matrix[i][j]), end=',')
    print()

X2PQ1_matrix = np.zeros([2, 2], dtype=complex)
X2PQ1_matrix[0][0] = X2PQ1_idleQ2_matrix[0][0]
X2PQ1_matrix[0][1] = X2PQ1_idleQ2_matrix[0][2]
X2PQ1_matrix[1][0] = X2PQ1_idleQ2_matrix[2][0]
X2PQ1_matrix[1][1] = X2PQ1_idleQ2_matrix[2][2]
print("\nX2PQ1_matrix:")
print(X2PQ1_matrix)

theta_g = (np.angle(X2PQ1_matrix[0][0])+np.angle(X2PQ1_matrix[1][1]))/2.0
phi = 2*np.arccos(np.real(X2PQ1_matrix[0][0]/np.exp(complex(0, 1)*theta_g)))
nx = np.imag(X2PQ1_matrix[0][1] /
             np.exp(complex(0, 1)*theta_g))/(-1)/np.sin(phi/2)
ny = np.real(X2PQ1_matrix[0][1] /
             np.exp(complex(0, 1)*theta_g))/(-1)/np.sin(phi/2)
nz = np.imag(X2PQ1_matrix[0][0] /
             np.exp(complex(0, 1)*theta_g))/(-1)/np.sin(phi/2)
print("theta_g=%.4f" % theta_g)
print("phi=%.4f" % phi)
print("nx=%.4f" % nx)
print("ny=%.4f" % ny)
print("nz=%.4f" % nz)
print("mod=%.4f" % (np.sqrt(nx**2+ny**2+nz**2)))

a = np.matmul(X2PQ1_matrix, X2PQ1_matrix.transpose().conjugate())
print("fedelity=%.5f" % ((np.abs(a[0][0]+a[1][1])/2.0)))
