# This file contains circuit's parameters.
import numpy as np
import constant as ct
import function as fun


class Circuit():
    # ====================================================================
    # Capacitor:
    # C_1: The left qubit's capacitor.
    # C_2: The middle coupler's capacitor.
    # C_3: The right qubit's capacitor.
    # C_12: The capacitor between left qubit and middle coupler.
    # C_23: The capacitor between right qubbit and middle coupler.
    # C_13: The capacitor between left qubit and right qubit.
    C_1 = 88E-15
    C_2 = 125E-15
    C_3 = 88E-15
    C_12 = 10E-15
    C_23 = 10E-15
    C_13 = 6E-16

    # Josephson junction critical current:
    # I_c1: The critical current of single junction of left qubit.
    # I_c2: The critical current of single junction of middle coupler.
    # I_c3: The critical current of single junction of right qubits.
    I_c1 = 1E-6
    I_c2 = 1E-6
    I_c3 = 1E-6

    # Remanence phase:
    # phi_r1: The remanence phase of left qubits's SQUID.
    # phi_r2: The remanence phase of middle coupler's SQUID.
    # phi_r3: The remanence phase of right qubit's SQUID.
    # PS: phi_ri/2/pi*PHI_ZERO=Phi_ri.
    phi_r1 = 0.1
    phi_r2 = 0.1
    phi_r3 = 0.1

    # Mutual inductance:
    # M_z_1: The mutual inductance between signal line 1 and left qubits's SQUID.
    # M_z_2: The mutual inductance between signal line 2 and middle coupler's SQUID.
    # M_z_3: The mutual inductance between signal line 3 and right qubits's SQUID.
    M_z_1 = 1E-12
    M_z_2 = 1E-12
    M_z_3 = 1E-12
    M_x_1 = 1E-12
    M_x_2 = 1E-12
    M_x_3 = 1E-12

    t_start = 0
    t_end = 20E-9
    t_piece = 0.05E-12
    t_piece_num = (t_end-t_start)/t_piece
    t_list = np.linspace(t_start, t_end, t_piece_num+1)
    signal_1 = 0
    signal_2 = 0
    signal_3 = 0
    operator_order_num = 3
    # ====================================================================

    def __init__(self):
        # Energy of electric charge:
        # E_c1: Energy of electric charge of left qubit.
        # E_c2: Energy of electric charge of middle coupler.
        # E_c3: Energy of electric charge of right qubit.
        # E_c12: Energy of electric charge of the capacitor between left qubit and middle coupler.
        # E_c23: Energy of electric charge of the capacitor between right qubit and middle coupler.
        # E_c13: Energy of electric charge of the capacitor between left qubit and right qubit.
        self.E_c1 = 0.5*ct.E*2/self.C_1
        self.E_c2 = 0.5*ct.E*2/self.C_2
        self.E_c3 = 0.5*ct.E*2/self.C_3
        self.E_c12 = 0.5*ct.E*2/self.C_12
        self.E_c23 = 0.5*ct.E*2/self.C_23
        self.E_c13 = 0.5*ct.E*2/self.C_13

        # Josephsn energy:
        # E_j1: Josephsn energy of left qubit's SQUID.
        # E_j2: Josephsn energy of middle coupler's SQUID.
        # E_j3: Josephsn energy of right qubit's SQUID.
        self.E_j1 = ct.PHI_ZERO*self.I_c1/np.pi
        self.E_j2 = ct.PHI_ZERO*self.I_c2/np.pi
        self.E_j3 = ct.PHI_ZERO*self.I_c3/np.pi

        # Quantum operator:
        # operator_identity: Identity operator with dimension of operator_order_num.
        # operator_phi_1: Phase operator of left qubit with dimension of operator_order_num.
        # operator_phi_2: Phase operator of middle coupler with dimension of operator_order_num.
        # operator_phi_3: Phase operator of right qubit with dimension of operator_order_num.
        # operator_n_1: Cooper pair number operator of left qubit with dimension of operator_order_num.
        # operator_n_2: Cooper pair number operator of middle coupler with dimension of operator_order_num.
        # operator_n_3: Cooper pair number operator of right qubit with dimension of operator_order_num.
        self.operator_identity = np.eye(self.operator_order_num)
        self.operator_phi_1 = np.power(2*self.E_c1/self.E_j1, 0.25)*np.kron(np.kron((fun.creation_operator_n(
            self.operator_order_num)+fun.annihilation_operator_n(self.operator_order_num)), self.operator_identity), self.operator_identity)
        self.operator_phi_2 = np.power(2*self.E_c2/self.E_j2, 0.25)*np.kron(np.kron(self.operator_identity, (fun.creation_operator_n(
            self.operator_order_num)+fun.annihilation_operator_n(self.operator_order_num))), self.operator_identity)
        self.operator_phi_3 = np.power(2*self.E_c3/self.E_j3, 0.25)*np.kron(np.kron(self.operator_identity, self.operator_identity), (fun.creation_operator_n(
            self.operator_order_num)+fun.annihilation_operator_n(self.operator_order_num)))
        self.operator_n_1=complex(0,0.5)*np.power(0.5*self.E_j1/self.E_c1, 0.25)*np.kron(np.kron((fun.creation_operator_n(
            self.operator_order_num)-fun.annihilation_operator_n(self.operator_order_num)), self.operator_identity), self.operator_identity)
        self.operator_n_2 = complex(0,0.5)*np.power(0.5*self.E_j2/self.E_c2, 0.25)*np.kron(np.kron(self.operator_identity, (fun.creation_operator_n(
            self.operator_order_num)-fun.annihilation_operator_n(self.operator_order_num))), self.operator_identity)
        self.operator_n_3 = complex(0,0.5)*np.power(0.5*self.E_j3/self.E_c3, 0.25)*np.kron(np.kron(self.operator_identity, self.operator_identity), (fun.creation_operator_n(
            self.operator_order_num)-fun.annihilation_operator_n(self.operator_order_num)))

    def Hamilton_calculation(self,n):
        Hamilton=
