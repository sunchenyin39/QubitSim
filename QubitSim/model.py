# This file contains circuit's model.
import numpy as np
import progressbar
import QubitSim.constant as ct
import QubitSim.function as fun
from matplotlib import pyplot as plt


class Circuit():
    def __init__(self):
        # ====================================================================
        # Capacitor:
        # C_1: The left qubit's capacitor.
        # C_2: The right qubit's capacitor.
        # C_3: The middle coupler's capacitor.
        # C_12: The capacitor between left qubit and right qubit.
        # C_23: The capacitor between right qubit and middle coupler.
        # C_13: The capacitor between left qubit and middle coupler.
        self.C_1 = 88.1E-15+1E-20
        self.C_2 = 88.1E-15+1E-20
        self.C_3 = 125.4E-15+1E-20
        self.C_12 = 6E-16+1E-20
        self.C_23 = 10.11E-15+1E-20
        self.C_13 = 10.11E-15+1E-20

        # L_off: Off inductance between low coupled notes.
        self.L_off = 1

        # Room temperature resistor of Josephson junction:
        # V_test: Voltage using to test room temperature resister.
        # R_1_1: The room temperature resistor of first junction of left qubit's DCSQUID.
        # R_1_2: The room temperature resistor of second junction of left qubit's DCSQUID.
        # R_2_1: The room temperature resistor of first junction of right qubits's DCSQUID.
        # R_2_2: The room temperature resistor of second junction of right qubits's DCSQUID.
        # R_3_1: The room temperature resistor of first junction of middle coupler's DCSQUID.
        # R_3_2: The room temperature resistor of second junction of middle coupler's DCSQUID.
        self.V_test = 2.8E-4
        self.R_1_1 = 18000
        self.R_1_2 = 18000
        self.R_2_1 = 18000
        self.R_2_2 = 18000
        self.R_3_1 = 3000
        self.R_3_2 = 2000

        # Remanence phase:
        # phi_r1: The remanence phase of left qubits's DCSQUID.
        # phi_r2: The remanence phase of right qubit's DCSQUID.
        # phi_r3: The remanence phase of middle coupler's DCSQUID.
        # PS: phi_ri/2/pi*PHI_ZERO=Phi_ri.
        self.phi_r1 = 0.0*np.pi
        self.phi_r2 = 0.12*np.pi
        self.phi_r3 = 0.39*np.pi

        # Mutual inductance:
        # M_z_1: The mutual inductance between signal line 1 and left qubit's DCSQUID.
        # M_z_2: The mutual inductance between signal line 2 and right qubit's DCSQUID.
        # M_z_3: The mutual inductance between signal line 3 and middle coupler's DCSQUID.
        # M_x_1: The mutual inductance between signal line 1 and left qubit's main loop.
        # M_x_2: The mutual inductance between signal line 2 and right qubit's main loop.
        # M_x_3: The mutual inductance between signal line 3 and middle coupler's main loop.
        self.M_z_1 = 1E-12
        self.M_z_2 = 1E-12
        self.M_z_3 = 1E-12
        self.M_x_1 = 1E-12
        self.M_x_2 = 1E-12
        self.M_x_3 = 1E-12

        # t_start: Starting time point.
        # t_end: Ending time point.
        # t_piece: Piece time.
        # operator_order_num.
        # trigonometric_function_expand_order_num.
        # exponent_function_expand_order_num.
        # picture_filename: Filename of picture to be drawed.
        # npy_filename: Filename of subspace quantum gate.
        # mode: The variable determines if the circuit needs phase gate.
        self.t_start = 0
        self.t_end = 20E-9
        self.t_piece = 1E-11
        self.operator_order_num = 4
        self.trigonometric_function_expand_order_num = 8
        self.exponent_function_expand_order_num = 15
        self.picture_filename = "picture.png"
        self.npy_filename = "gate.npy"
        self.mode = 0
        # ====================================================================

    def initial(self):
        # operator_order_num_change: Operator expanding order using to calculating H0.
        # t_piece_num: 2*Number of piece time.
        # t_list: Time list.
        # signal_1: Signal adding to left qubit's main loop.
        # signal_2: Signal adding to right qubit's main loop.
        # signal_3: Signal adding to middle coupler's main loop.
        # signal_1z: Signal adding to left qubit's DCSQUID.
        # signal_2z: Signal adding to right qubit's DCSQUID.
        # signal_3z: Signal adding to middle coupler's DCSQUID.
        self.operator_order_num_change = self.operator_order_num+5
        self.t_piece_num = int(
            np.round(2*(self.t_end-self.t_start)/self.t_piece))
        self.t_list = np.linspace(self.t_start, self.t_end, self.t_piece_num+1)
        self.signal_1 = np.ones(self.t_piece_num+1)*0
        self.signal_2 = np.ones(self.t_piece_num+1)*0
        self.signal_3 = np.ones(self.t_piece_num+1)*0
        self.signal_1z = np.ones(self.t_piece_num+1)*0
        self.signal_2z = np.ones(self.t_piece_num+1)*0
        self.signal_3z = np.ones(self.t_piece_num+1)*0

        # Josephson junction critical current:
        # I_c1_1: The critical current of first junction of left qubit's DCSQUID.
        # I_c1_2: The critical current of second junction of left qubit's DCSQUID.
        # I_c2_1: The critical current of first junction of right qubit's DCSQUID.
        # I_c2_2: The critical current of second junction of right qubit's DCSQUID.
        # I_c3_1: The critical current of first junction of middle coupler's DCSQUID.
        # I_c3_2: The critical current of second junction of middle coupler's DCSQUID.
        self.I_c1_1 = self.V_test/self.R_1_1
        self.I_c1_2 = self.V_test/self.R_1_2
        self.I_c2_1 = self.V_test/self.R_2_1
        self.I_c2_2 = self.V_test/self.R_2_2
        self.I_c3_1 = self.V_test/self.R_3_1
        self.I_c3_2 = self.V_test/self.R_3_2

        # M_C: Capacitor matrix.
        # M_Ec: Matrix of energy of electric charge.
        self.M_C = np.array([[self.C_1+self.C_12+self.C_13, -self.C_12, -self.C_13], [
            -self.C_12, self.C_2+self.C_12+self.C_23, -self.C_23], [-self.C_13, -self.C_23, self.C_3+self.C_13+self.C_23]])
        self.M_Ec = 0.5*ct.E**2*np.linalg.pinv(self.M_C)

        # Energy of electric charge:
        # E_c1: Energy of electric charge of left qubit.
        # E_c2: Energy of electric charge of right qubit.
        # E_c3: Energy of electric charge of middle coupler.
        # E_c12: Energy of electric charge of the capacitor between left qubit and right qubit.
        # E_c23: Energy of electric charge of the capacitor between right qubit and middle coupler.
        # E_c13: Energy of electric charge of the capacitor between left qubit and middle coupler.
        self.E_c1 = self.M_Ec[0][0]
        self.E_c2 = self.M_Ec[1][1]
        self.E_c3 = self.M_Ec[2][2]
        self.E_c12 = self.M_Ec[0][1]
        self.E_c23 = self.M_Ec[1][2]
        self.E_c13 = self.M_Ec[0][2]

        # operator_identity: Identity operator with dimension of operator_order_num.
        self.operator_identity = np.eye(self.operator_order_num)

    def M_L_generator(self, signal_1z, signal_2z, signal_3z):
        """The function calculationg inductance matrix.

        Args:
            signal_1z(float): Signal_1z's value.
            signal_2z(float): Signal_2z's value.
            signal_3z(float): Signal_3z's value.
        Returns:
            np.array: Inductance matrix.
        """
        phi_list = [self.phi_r1+signal_1z*np.pi, self.phi_r2 +
                    signal_2z*np.pi, self.phi_r3+signal_3z*np.pi]
        Ic_1 = [self.I_c1_1, self.I_c2_1, self.I_c3_1]
        Ic_2 = [self.I_c1_2, self.I_c2_2, self.I_c3_2]
        M_L = np.ones([3, 3])*self.L_off
        for i in range(3):
            M_L[i][i] = ct.PHI_ZERO/2/np.pi / \
                np.sqrt(Ic_1[i]**2+Ic_2[i]**2+2*Ic_1[i]
                        * Ic_2[i]*np.cos(2*phi_list[i]))
        return M_L

    def M_Ej_generator(self, signal_z1, signal_z2, signal_z3):
        """The function calculationg Josephson energy matrix.

        Args:
            signal_1z(float): Signal_1z's value.
            signal_2z(float): Signal_2z's value.
            signal_3z(float): Signal_3z's value.

        Returns:
            np.array: Josephson energy matrix.
        """
        M_Ej_0 = ct.H**2/(4*np.pi**2*4*ct.E**2) / \
            self.M_L_generator(signal_z1, signal_z2, signal_z3)
        M_Ej = np.zeros([3, 3])
        for i in range(3):
            for j in range(3):
                if (i == j):
                    M_Ej[i][j] = np.sum(M_Ej_0[:, i])
                else:
                    M_Ej[i][j] = -M_Ej_0[i][j]
        return M_Ej

    def operator_phi_generator(self, E_c, E_j, operator_order_num):
        """The function generating phase operator with order of operator_order_num.

        Args:
            E_c (float): Electric energy.
            E_j (float): Josephson energy.
            operator_order_num (int): Expanding order of operator. 

        Returns:
            np.array: Returned phase operator.
        """
        return np.power(2*E_c/E_j, 0.25)*(fun.creation_operator_n(operator_order_num)+fun.annihilation_operator_n(operator_order_num))

    def operator_n_generator(self, E_c, E_j, operator_order_num):
        """The function generating phase operator with order of operator_order_num.

        Args:
            E_c (float): Electric energy.
            E_j (float): Josephson energy.
            operator_order_num (int): Expanding order of operator. 

        Returns:
            np.array: Returned phase operator.
        """
        return complex(0, 0.5)*np.power(0.5*E_j/E_c, 0.25)*(fun.creation_operator_n(operator_order_num)-fun.annihilation_operator_n(operator_order_num))

    def Hamiltonian_calculation(self, signal_1, signal_2, signal_3, signal_1_aux, signal_2_aux, signal_3_aux, signal_z1, signal_z2, signal_z3):
        """The function calculating Hamiltonian.


        Args:
            signal_1(float): Middle value of signal_1.
            signal_2(float): Middle value of signal_2.
            signal_3(float): Middle value of signal_3.
            signal_1_aux(float): Auxiliary value of signal_1.
            signal_2_aux(float): Auxiliary value of signal_2.
            signal_3_aux(float): Auxiliary value of signal_3.
            signal_1z(float): Signal_1z's value.
            signal_2z(float): Signal_2z's value.
            signal_3z(float): Signal_3z's value.

        Returns:
            np.array: The n'st time piece's Hamiltonian.
        """

        # M_Ej: Matrix of Josephson energy.
        # E_j1: Josephsn energy of left qubit's DCSQUID.
        # E_j2: Josephsn energy of right qubit's DCSQUID.
        # E_j3: Josephsn energy of middle coupler's DCSQUID.
        M_Ej = self.M_Ej_generator(signal_z1, signal_z2, signal_z3)
        E_j1 = M_Ej[0][0]
        E_j2 = M_Ej[1][1]
        E_j3 = M_Ej[2][2]

        # Quantum operator:
        # operator_phi_1: Phase operator of left qubit with dimension of operator_order_num**3.
        # operator_phi_2: Phase operator of right qubit with dimension of operator_order_num**3.
        # operator_phi_3: Phase operator of middle coupler with dimension of operator_order_num**3.
        # operator_n_1: Cooper pair number operator of left qubit with dimension of operator_order_num**3.
        # operator_n_2: Cooper pair number operator of right qubit with dimension of operator_order_num**3.
        # operator_n_3: Cooper pair number operator of middle coupler with dimension of operator_order_num**3.
        operator_phi_1 = np.power(2*self.E_c1/E_j1, 0.25)*np.kron(np.kron((fun.creation_operator_n(
            self.operator_order_num)+fun.annihilation_operator_n(self.operator_order_num)), self.operator_identity), self.operator_identity)
        operator_phi_2 = np.power(2*self.E_c2/E_j2, 0.25)*np.kron(np.kron(self.operator_identity, (fun.creation_operator_n(
            self.operator_order_num)+fun.annihilation_operator_n(self.operator_order_num))), self.operator_identity)
        operator_phi_3 = np.power(2*self.E_c3/E_j3, 0.25)*np.kron(np.kron(self.operator_identity, self.operator_identity), (fun.creation_operator_n(
            self.operator_order_num)+fun.annihilation_operator_n(self.operator_order_num)))
        operator_n_1 = complex(0, 0.5)*np.power(0.5*E_j1/self.E_c1, 0.25)*np.kron(np.kron((fun.creation_operator_n(
            self.operator_order_num)-fun.annihilation_operator_n(self.operator_order_num)), self.operator_identity), self.operator_identity)
        operator_n_2 = complex(0, 0.5)*np.power(0.5*E_j2/self.E_c2, 0.25)*np.kron(np.kron(self.operator_identity, (fun.creation_operator_n(
            self.operator_order_num)-fun.annihilation_operator_n(self.operator_order_num))), self.operator_identity)
        operator_n_3 = complex(0, 0.5)*np.power(0.5*E_j3/self.E_c3, 0.25)*np.kron(np.kron(self.operator_identity, self.operator_identity), (fun.creation_operator_n(
            self.operator_order_num)-fun.annihilation_operator_n(self.operator_order_num)))

        Y = complex(0, 1)*(fun.annihilation_operator_n(self.operator_order_num_change) -
                           fun.creation_operator_n(self.operator_order_num_change))/np.sqrt(2)
        # Adding left qubit's energy to returned Hamiltionian.
        Hamiltonian_temp = 0.5*np.sqrt(8*self.E_c1*E_j1)*np.matmul((Y-signal_1_aux*np.eye(self.operator_order_num_change)), (Y-signal_1_aux*np.eye(self.operator_order_num_change)))-E_j1*fun.cos_matrix_n(self.operator_phi_generator(
            self.E_c1, E_j1, self.operator_order_num_change)-np.power(8*self.E_c1/E_j1, 0.25)*signal_1*np.eye(self.operator_order_num_change), self.trigonometric_function_expand_order_num)+(E_j1-0.5*np.sqrt(8*E_j1*self.E_c1))*np.eye(self.operator_order_num_change)
        Hamiltonian_temp = Hamiltonian_temp[0:self.operator_order_num,
                                            0:self.operator_order_num]
        Hamiltonian = np.kron(np.kron(Hamiltonian_temp, self.operator_identity),
                              self.operator_identity)
        # Adding right qubit's energy to returned Hamiltionian.
        Hamiltonian_temp = 0.5*np.sqrt(8*self.E_c2*E_j2)*np.matmul((Y-signal_2_aux*np.eye(self.operator_order_num_change)), (Y-signal_2_aux*np.eye(self.operator_order_num_change)))-E_j2*fun.cos_matrix_n(self.operator_phi_generator(
            self.E_c2, E_j2, self.operator_order_num_change)-np.power(8*self.E_c2/E_j2, 0.25)*signal_2*np.eye(self.operator_order_num_change), self.trigonometric_function_expand_order_num)+(E_j2-0.5*np.sqrt(8*E_j2*self.E_c2))*np.eye(self.operator_order_num_change)
        Hamiltonian_temp = Hamiltonian_temp[0:self.operator_order_num,
                                            0:self.operator_order_num]
        Hamiltonian = Hamiltonian + \
            np.kron(np.kron(self.operator_identity, Hamiltonian_temp),
                    self.operator_identity)
        # Adding middle coupler's energy to returned Hamiltionian.
        Hamiltonian_temp = 0.5*np.sqrt(8*self.E_c3*E_j3)*np.matmul((Y-signal_3_aux*np.eye(self.operator_order_num_change)), (Y-signal_3_aux*np.eye(self.operator_order_num_change)))-E_j3*fun.cos_matrix_n(self.operator_phi_generator(
            self.E_c3, E_j3, self.operator_order_num_change)-np.power(8*self.E_c3/E_j3, 0.25)*signal_3*np.eye(self.operator_order_num_change), self.trigonometric_function_expand_order_num)+(E_j3-0.5*np.sqrt(8*E_j3*self.E_c3))*np.eye(self.operator_order_num_change)
        Hamiltonian_temp = Hamiltonian_temp[0:self.operator_order_num,
                                            0:self.operator_order_num]
        Hamiltonian = Hamiltonian + \
            np.kron(np.kron(self.operator_identity,
                    self.operator_identity), Hamiltonian_temp)

        # Hamiltonian_interact_electric: Electric interaction hamiltonian.
        # Hamiltonian_interact_magnetic: Magnetic interaction hamiltonian.
        Hamiltonian_interact_electric = 8*self.E_c12*np.matmul(operator_n_1, operator_n_2)+8*self.E_c23*np.matmul(
            operator_n_2, operator_n_3)+8*self.E_c13*np.matmul(operator_n_1, operator_n_3)
        Hamiltonian_interact_magnetic = M_Ej[0][1]*np.matmul(operator_phi_1, operator_phi_2)+M_Ej[0][2]*np.matmul(
            operator_phi_1, operator_phi_3)+M_Ej[1][2]*np.matmul(operator_phi_2, operator_phi_3)
        # Adding interaction Hamiltonian to Hamiltonian.
        Hamiltonian = Hamiltonian+Hamiltonian_interact_electric+Hamiltonian_interact_magnetic

        return Hamiltonian

    def time_evolution_operator_calculation(self, n):
        """The function calculating the n'st time piece's time evolution operator.

        Args:
            n (int): The n'st time piece.

        Returns:
            np.array: The n'st time piece's time evolution operator.
        """
        t_piece = self.t_piece*1E9
        Hamiltonian_middle = self.Hamiltonian_calculation(
            self.signal_1[2*n-1], self.signal_2[2*n-1], self.signal_3[2*n-1], 0, 0, 0, self.signal_1z[2*n-1], self.signal_2z[2*n-1], self.signal_3z[2*n-1])/ct.H/1E9
        Hamiltonian_left = self.Hamiltonian_calculation(
            self.signal_1[2*n-1], self.signal_2[2*n-1], self.signal_3[2*n-1], self.signal_1[2*n-2], self.signal_2[2*n-2], self.signal_3[2*n-2], self.signal_1z[2*n-2], self.signal_2z[2*n-2], self.signal_3z[2*n-2])/ct.H/1E9
        Hamiltonian_right = self.Hamiltonian_calculation(
            self.signal_1[2*n-1], self.signal_2[2*n-1], self.signal_3[2*n-1], self.signal_1[2*n], self.signal_2[2*n], self.signal_3[2*n], self.signal_1z[2*n], self.signal_2z[2*n], self.signal_3z[2*n])/ct.H/1E9
        Hamiltonian_I = (Hamiltonian_right-Hamiltonian_left)/t_piece
        Hamiltonian_II = 4*(Hamiltonian_right+Hamiltonian_left -
                            2*Hamiltonian_middle)/(t_piece**2)
        Hamiltonian_I0 = np.matmul(
            Hamiltonian_middle, Hamiltonian_I)-np.matmul(Hamiltonian_I, Hamiltonian_middle)

        time_evolution_operator = fun.exp_matrix_n(-2*np.pi*complex(0, 1)*(Hamiltonian_middle*t_piece+1/24*Hamiltonian_II *
                                                   t_piece**3)+4*np.pi**2/12*Hamiltonian_I0*t_piece**3, self.exponent_function_expand_order_num)

        return time_evolution_operator

    def transformational_matrix_generator(self):
        """The function generating transformational matrix converting bare bases to dressed bases.

        Returns:
            (np.array,np.array): The first return is eigenvalue list and the second return is featurevector matrix.
        """
        H_0 = self.Hamiltonian_calculation(0, 0, 0, 0, 0, 0, 0, 0, 0)
        eigenvalue, featurevector_temp = np.linalg.eig(H_0)
        eigenvalue = np.real(eigenvalue)
        featurevector = np.zeros(
            [self.operator_order_num**3, self.operator_order_num**3], dtype=complex)
        sort_index_list = np.argsort(eigenvalue)
        for i in range(len(sort_index_list)):
            featurevector[:, i] = featurevector_temp[:, sort_index_list[i]]
        eigenvalue = np.sort(eigenvalue)
        return (eigenvalue, featurevector)

    def dressed_state_subspace_phase_process(self):
        """The function converting time evolution operator from bare bases to dressed bases, subspace processing and phase reset processing.

        Returns:
            (np.array,np.array): The first return is the time evolution operator in dressed bases and the 
            second return is the sub time evolution operator in dressed bases of kets 00 01 10 11. 
        """
        time_evolution_operator_dressed = np.matmul(np.linalg.inv(
            self.dressed_featurevector), np.matmul(self.time_evolution_operator, self.dressed_featurevector))
        index00 = self.dressed_state_index_find([0, 0, 0])
        index01 = self.dressed_state_index_find([0, 1, 0])
        index10 = self.dressed_state_index_find([1, 0, 0])
        index11 = self.dressed_state_index_find([1, 1, 0])

        index_list = [index00, index01, index10, index11]
        time_evolution_operator_dressed_sub = np.zeros(
            [len(index_list), len(index_list)], dtype=complex)
        for i in range(len(index_list)):
            for j in range(len(index_list)):
                time_evolution_operator_dressed_sub[i][j] = time_evolution_operator_dressed[index_list[i]][index_list[j]]
        phase_gate = np.array([[1.0, 0, 0, 0], [0, np.exp(2*np.pi*complex(0, 1)/ct.H*self.Eb*self.t_end), 0, 0], [0, 0, np.exp(
            2*np.pi*complex(0, 1)/ct.H*self.Ea*self.t_end), 0], [0, 0, 0, np.exp(2*np.pi*complex(0, 1)/ct.H*(self.Ea+self.Eb)*self.t_end)]])
        if self.mode == 1:
            time_evolution_operator_dressed_sub = np.matmul(
                time_evolution_operator_dressed_sub, phase_gate)
        return (time_evolution_operator_dressed, time_evolution_operator_dressed_sub)

    def dressed_state_index_find(self, bare_state_list):
        """The function finding the corresponding dress state's index according to the bare state's tag.

        Args:
            bare_state_list (list[int]): Bare state tag.

        Returns:
            int: The index of dressed state in dressed_featurevector.
        """
        bare_state_index = 0
        for i in range(3):
            bare_state_index = bare_state_index+bare_state_list[i] * \
                self.operator_order_num**(2-i)
        return np.argmax(np.abs(self.dressed_featurevector[bare_state_index, :]))

    def dataprocess(self, filename="picture.png"):
        """Dataprocess function used to plot some matrix elements' modulus changing over time.

        Args:
            filename (str, optional): Filename of picture for saving. Defaults to "picture.png".
        """
        namelist = ['00', '01', '10', '11', '20', '02']
        state_start = [0, 0, 3, 3, 3]
        state_evolution = [1, 2, 3, 4, 5]
        curve_lists = np.zeros(
            [len(state_start), len(self.time_evolution_operator_path)])
        t_list = np.linspace(self.t_start, self.t_end, int(np.round(
            (self.t_end-self.t_start)/self.t_piece))+1)*1E9
        index00 = self.dressed_state_index_find([0, 0, 0])
        index01 = self.dressed_state_index_find([0, 1, 0])
        index10 = self.dressed_state_index_find([1, 0, 0])
        index11 = self.dressed_state_index_find([1, 1, 0])
        index20 = self.dressed_state_index_find([2, 0, 0])
        index02 = self.dressed_state_index_find([0, 2, 0])
        index_list = [index00, index01, index10, index11, index20, index02]
        for i in range(len(state_start)):
            for j in range(len(self.time_evolution_operator_path)):
                curve_lists[i][j] = np.abs(
                    self.time_evolution_operator_path[j][index_list[state_start[i]]][index_list[state_evolution[i]]])**2
        plt.figure(figsize=[19.2, 12])
        ax = plt.subplot(2, 3, 1)
        ax.plot(t_list, curve_lists[0])
        ax.set_title(
            "(Q1,Q2) P"+namelist[state_start[0]]+" to P"+namelist[state_evolution[0]])
        ax.set_xlabel("t/ns")
        ax = plt.subplot(2, 3, 2)
        ax.plot(t_list, curve_lists[1])
        ax.set_title(
            "(Q1,Q2) P"+namelist[state_start[1]]+" to P"+namelist[state_evolution[1]])
        ax.set_xlabel("t/ns")
        ax = plt.subplot(2, 3, 3)
        ax.plot(t_list, curve_lists[2])
        ax.set_title(
            "(Q1,Q2) P"+namelist[state_start[2]]+" to P"+namelist[state_evolution[2]])
        ax.set_xlabel("t/ns")
        ax = plt.subplot(2, 3, 4)
        ax.plot(t_list, curve_lists[3])
        ax.set_title(
            "(Q1,Q2) P"+namelist[state_start[3]]+" to P"+namelist[state_evolution[3]])
        ax.set_xlabel("t/ns")
        ax = plt.subplot(2, 3, 5)
        ax.plot(t_list, curve_lists[4])
        ax.set_title(
            "(Q1,Q2) P"+namelist[state_start[4]]+" to P"+namelist[state_evolution[4]])
        ax.set_xlabel("t/ns")
        plt.tight_layout()
        plt.savefig(fname=filename)

    def run(self):
        # 1.Getting transformational matrix converting bare bases to dressed bases.
        # dressed_eigenvalue: Dressed states' energy eigenvalue.
        # dressed_featurevector: Transformational matrix converting bare bases to dressed bases
        self.dressed_eigenvalue, self.dressed_featurevector = self.transformational_matrix_generator()
        self.Ea = (self.dressed_eigenvalue-min(self.dressed_eigenvalue))[1]
        self.Eb = (self.dressed_eigenvalue-min(self.dressed_eigenvalue))[2]

        # 2.Simulation calculating the whole time evolution operator.
        p = progressbar.ProgressBar()
        self.time_evolution_operator = np.eye(self.operator_order_num**3)
        self.time_evolution_operator_path = []
        self.time_evolution_operator_path.append(np.matmul(np.linalg.inv(
            self.dressed_featurevector), np.matmul(self.time_evolution_operator, self.dressed_featurevector)))
        print("Calculating the whole time evolution operator:")
        for i in p(range(int(self.t_piece_num/2))):
            self.time_evolution_operator = np.matmul(
                self.time_evolution_operator_calculation(i+1), self.time_evolution_operator)
            self.time_evolution_operator_path.append(np.matmul(np.linalg.inv(
                self.dressed_featurevector), np.matmul(self.time_evolution_operator, self.dressed_featurevector)))

        # 3.Dressed state process, subspace process, phase process.
        self.time_evolution_operator_dressed, self.time_evolution_operator_dressed_sub = self.dressed_state_subspace_phase_process()

        # 4.Data process.
        self.dataprocess(self.picture_filename)

        # 5.Subspace gate saving.
        np.save(self.npy_filename, self.time_evolution_operator_dressed_sub)
