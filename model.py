# This file contains circuit's model.
import numpy as np
import progressbar
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
    C_1 = 88.1E-15
    C_2 = 125.4E-15
    C_3 = 88E-15
    C_12 = 10.11E-15
    C_23 = 10.11E-15
    C_13 = 6E-16

    # Room temperature resistor of Josephson junction:
    # V_test: Voltage using to test room temperature resister.
    # R_1_1: The room temperature resistor of first junction of left qubit's DCSQUID.
    # R_1_2: The room temperature resistor of second junction of left qubit's DCSQUID.
    # R_2_1: The room temperature resistor of first junction of middle coupler's DCSQUID.
    # R_2_2: The room temperature resistor of second junction of middle coupler's DCSQUID.
    # R_3_1: The room temperature resistor of first junction of right qubits's DCSQUID.
    # R_3_2: The room temperature resistor of second junction of right qubits's DCSQUID.
    V_test = 2.8E-4
    R_1_1 = 18000
    R_1_2 = 18000
    R_2_1 = 4000
    R_2_2 = 2000
    R_3_1 = 18000
    R_3_2 = 18000

    # Remanence phase:
    # phi_r1: The remanence phase of left qubits's DCSQUID.
    # phi_r2: The remanence phase of middle coupler's DCSQUID.
    # phi_r3: The remanence phase of right qubit's DCSQUID.
    # PS: phi_ri/2/pi*PHI_ZERO=Phi_ri.
    phi_r1 = 0.0
    phi_r2 = 0.12
    phi_r3 = 0.39

    # Mutual inductance:
    # M_z_1: The mutual inductance between signal line 1 and left qubits's DCSQUID.
    # M_z_2: The mutual inductance between signal line 2 and middle coupler's DCSQUID.
    # M_z_3: The mutual inductance between signal line 3 and right qubits's DCSQUID.
    M_z_1 = 1E-12
    M_z_2 = 1E-12
    M_z_3 = 1E-12
    M_x_1 = 1E-12
    M_x_2 = 1E-12
    M_x_3 = 1E-12

    # t_start: Starting time point.
    # t_end: Ending time point.
    # t_piece: Piece time.
    # t_piece_num: Number of piece time.
    # t_list: Time list.
    # signal_1: Signal adding to left qubit's DCSQUID.
    # signal_2: Signal adding to middle coupler's DCSQUID.
    # signal_3: Signal adding to right qubit's DCSQUID.
    # operator_order_num.
    # trigonometric_function_expand_order_num.
    # exponent_function_expand_order_num.
    t_start = 0
    t_end = 20E-9
    t_piece = 1E-12
    t_piece_num = int((t_end-t_start)/t_piece)
    t_list = np.linspace(t_start, t_end, t_piece_num+1)
    signal_1 = np.zeros(t_piece_num+1)
    signal_2 = np.zeros(t_piece_num+1)
    signal_3 = np.zeros(t_piece_num+1)
    operator_order_num = 3
    trigonometric_function_expand_order_num = 4
    exponent_function_expand_order_num = 100
    # ====================================================================

    def __init__(self):
        # Josephson junction critical current:
        # I_c1_1: The critical current of first junction of left qubit's DCSQUID.
        # I_c1_2: The critical current of second junction of left qubit's DCSQUID.
        # I_c2_1: The critical current of first junction of middle coupler's DCSQUID.
        # I_c2_2: The critical current of second junction of middle coupler's DCSQUID.
        # I_c3_1: The critical current of first junction of right qubits's DCSQUID.
        # I_c3_2: The critical current of second junction of right qubits's DCSQUID.
        self.I_c1_1 = self.V_test/self.R_1_1
        self.I_c1_2 = self.V_test/self.R_1_2
        self.I_c2_1 = self.V_test/self.R_2_1
        self.I_c2_2 = self.V_test/self.R_2_2
        self.I_c3_1 = self.V_test/self.R_3_1
        self.I_c3_2 = self.V_test/self.R_3_2

        # Energy of electric charge:
        # E_c1: Energy of electric charge of left qubit.
        # E_c2: Energy of electric charge of middle coupler.
        # E_c3: Energy of electric charge of right qubit.
        # E_c12: Energy of electric charge of the capacitor between left qubit and middle coupler.
        # E_c23: Energy of electric charge of the capacitor between right qubit and middle coupler.
        # E_c13: Energy of electric charge of the capacitor between left qubit and right qubit.
        self.E_c1 = 0.5*ct.E**2/self.C_1
        self.E_c2 = 0.5*ct.E**2/self.C_2
        self.E_c3 = 0.5*ct.E**2/self.C_3
        self.E_c12 = 0.5*ct.E**2/self.C_12
        self.E_c23 = 0.5*ct.E**2/self.C_23
        self.E_c13 = 0.5*ct.E**2/self.C_13

        # Josephsn energy:
        # E_j1_1: Josephsn energy of left qubit's DCSQUID's first junction.
        # E_j1_2: Josephsn energy of left qubit's DCSQUID's second junction.
        # E_j2_1: Josephsn energy of middle coupler's DCSQUID's first junction.
        # E_j2_2: Josephsn energy of middle coupler's DCSQUID's second junction.
        # E_j3_1: Josephsn energy of right qubit's DCSQUID's first junction.
        # E_j3_2: Josephsn energy of right qubit's DCSQUID's second junction.
        self.E_j1_1 = 0.5*ct.PHI_ZERO*self.I_c1_1/np.pi
        self.E_j1_2 = 0.5*ct.PHI_ZERO*self.I_c1_2/np.pi
        self.E_j2_1 = 0.5*ct.PHI_ZERO*self.I_c2_1/np.pi
        self.E_j2_2 = 0.5*ct.PHI_ZERO*self.I_c2_2/np.pi
        self.E_j3_1 = 0.5*ct.PHI_ZERO*self.I_c3_1/np.pi
        self.E_j3_2 = 0.5*ct.PHI_ZERO*self.I_c3_2/np.pi
        self.E_j1 = (self.E_j1_1+self.E_j1_2)*np.cos(self.phi_r1)
        self.E_j2 = (self.E_j2_1+self.E_j2_2)*np.cos(self.phi_r2)
        self.E_j3 = (self.E_j3_1+self.E_j3_2)*np.cos(self.phi_r3)

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
        self.operator_n_1 = complex(0, 0.5)*np.power(0.5*self.E_j1/self.E_c1, 0.25)*np.kron(np.kron((fun.creation_operator_n(
            self.operator_order_num)-fun.annihilation_operator_n(self.operator_order_num)), self.operator_identity), self.operator_identity)
        self.operator_n_2 = complex(0, 0.5)*np.power(0.5*self.E_j2/self.E_c2, 0.25)*np.kron(np.kron(self.operator_identity, (fun.creation_operator_n(
            self.operator_order_num)-fun.annihilation_operator_n(self.operator_order_num))), self.operator_identity)
        self.operator_n_3 = complex(0, 0.5)*np.power(0.5*self.E_j3/self.E_c3, 0.25)*np.kron(np.kron(self.operator_identity, self.operator_identity), (fun.creation_operator_n(
            self.operator_order_num)-fun.annihilation_operator_n(self.operator_order_num)))

        # Electric charge energy Hamiltonian.
        self.Hamiltonian_electric = 4*self.E_c1*np.matmul(self.operator_n_1, self.operator_n_1)+4*self.E_c2*np.matmul(
            self.operator_n_2, self.operator_n_2)+4*self.E_c3*np.matmul(self.operator_n_3, self.operator_n_3)+8*self.E_c12*np.matmul(self.operator_n_1, self.operator_n_2)+8*self.E_c23*np.matmul(self.operator_n_2, self.operator_n_3)+8*self.E_c13*np.matmul(self.operator_n_1, self.operator_n_3)

    def Hamiltonian_calculation(self, n):
        """The function calculating the n'st time piece's Hamiltonian. If n equals to zero, this function calculating
        

        Args:
            n (int): The n'st time piece.

        Returns:
            np.array: The n'st time piece's Hamiltonian.
        """
        # signal_1_n: The approximate value of signal_1's n'st time piece.
        # signal_2_n: The approximate value of signal_2's n'st time piece.
        # signal_3_n: The approximate value of signal_3's n'st time piece.
        if n == 0:
            signal_1_n = self.signal_1[0]
            signal_2_n = self.signal_2[0]
            signal_3_n = self.signal_3[0]
        else:
            signal_1_n = 0.5*(self.signal_1[n-1]+self.signal_1[n])
            signal_2_n = 0.5*(self.signal_2[n-1]+self.signal_2[n])
            signal_3_n = 0.5*(self.signal_3[n-1]+self.signal_3[n])

        # Adding left qubit's Josephson energy to returned Hamiltionian.
        Hamiltonian = self.Hamiltonian_electric - (self.E_j1_1+self.E_j1_2)*np.cos(self.phi_r1+np.pi*self.M_z_1*signal_1_n/ct.PHI_ZERO)*fun.cos_alpha_matrix_n(2*np.pi*self.M_x_1*signal_1_n/ct.PHI_ZERO, self.operator_phi_1, self.trigonometric_function_expand_order_num) - (
            self.E_j1_2-self.E_j1_1)*np.sin(self.phi_r1+np.pi*self.M_z_1*signal_1_n/ct.PHI_ZERO)*fun.sin_alpha_matrix_n(2 * np.pi*self.M_x_1*signal_1_n/ct.PHI_ZERO, self.operator_phi_1, self.trigonometric_function_expand_order_num)
        # Adding middle coupler's Josephson energy to returned Hamiltionian.
        Hamiltonian = Hamiltonian - (self.E_j2_1+self.E_j2_2)*np.cos(self.phi_r2+np.pi*self.M_z_2*signal_2_n/ct.PHI_ZERO)*fun.cos_alpha_matrix_n(2*np.pi*self.M_x_2*signal_2_n/ct.PHI_ZERO, self.operator_phi_2, self.trigonometric_function_expand_order_num)-(
            self.E_j2_2-self.E_j2_1)*np.sin(self.phi_r2+np.pi*self.M_z_2*signal_2_n/ct.PHI_ZERO)*fun.sin_alpha_matrix_n(2*np.pi*self.M_x_2*signal_2_n/ct.PHI_ZERO, self.operator_phi_2, self.trigonometric_function_expand_order_num)
        # Adding right qubit's Josephson energy to returned Hamiltionian.
        Hamiltonian = Hamiltonian - (self.E_j3_1+self.E_j3_2)*np.cos(self.phi_r3+np.pi*self.M_z_3*signal_3_n/ct.PHI_ZERO)*fun.cos_alpha_matrix_n(2*np.pi*self.M_x_3*signal_3_n/ct.PHI_ZERO, self.operator_phi_3, self.trigonometric_function_expand_order_num)-(
            self.E_j3_2-self.E_j3_1)*np.sin(self.phi_r3+np.pi*self.M_z_3*signal_3_n/ct.PHI_ZERO)*fun.sin_alpha_matrix_n(2*np.pi*self.M_x_3*signal_3_n/ct.PHI_ZERO, self.operator_phi_3, self.trigonometric_function_expand_order_num)

        return Hamiltonian

    def time_evolution_operator_calculation(self, n):
        """The function calculating the n'st time piece's time evolution operator.

        Args:
            n (int): The n'st time piece.

        Returns:
            np.array: The n'st time piece's time evolution operator.
        """
        time_evolution_operator = fun.exp_matrix_n(-2*np.pi*complex(0, 1)*self.Hamiltonian_calculation(
            n)*self.t_piece/ct.H, self.exponent_function_expand_order_num)
        return time_evolution_operator

    def time_evolution_operator_calculation_all(self):
        """The function calculating the whole time evolution operator.

        Returns:
            np.array: The whole time evolution operator.
        """
        p = progressbar.ProgressBar()
        time_evolution_operator = np.eye(self.operator_order_num**3)
        print("Calculating the whole time evolution operator:")
        for i in p(range(self.t_piece_num)):
            time_evolution_operator = np.matmul(
                self.time_evolution_operator_calculation(i+1), time_evolution_operator)
        return time_evolution_operator
