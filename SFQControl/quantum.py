import numpy as np


def UGgenerator(USFQ, UFR, Subsequence, times, operator_order_num, dressed_eigenvalue, dressed_featurevector, T_SFQclock):
    # 1.Generate the whole time evolution operator.
    UG = np.eye(USFQ.shape[0], dtype=complex)
    for i in range(len(Subsequence)):
        if Subsequence[i] == 0:
            UG = np.matmul(UFR, UG)
        else:
            UG = np.matmul(USFQ, UG)
    UG_temp = UG
    for i in range(times-1):
        UG = np.matmul(UG_temp, UG)
    # 2.Converting to dressed states.
    UG = np.matmul(np.linalg.inv(dressed_featurevector),
                   np.matmul(UG, dressed_featurevector))
    # 3.Subgate
    index00 = dressed_state_index_find(
        [0, 0, 0], operator_order_num, dressed_featurevector)
    index01 = dressed_state_index_find(
        [0, 1, 0], operator_order_num, dressed_featurevector)
    index10 = dressed_state_index_find(
        [1, 0, 0], operator_order_num, dressed_featurevector)
    index11 = dressed_state_index_find(
        [1, 1, 0], operator_order_num, dressed_featurevector)
    index_list = [index00, index01, index10, index11]
    UG_sub = np.zeros([len(index_list), len(index_list)], dtype=complex)
    for i in range(len(index_list)):
        for j in range(len(index_list)):
            UG_sub[i][j] = UG[index_list[i]][index_list[j]]
    # 4.Phase processing
    H = 6.62E-34
    t_end = times*len(Subsequence)*T_SFQclock
    E_00 = dressed_eigenvalue[0]
    E_01 = dressed_eigenvalue[1]
    E_10 = dressed_eigenvalue[2]
    E_11 = E_10+E_01-E_00
    phase_gate = np.array([[np.exp(2*np.pi*complex(0, 1)/H*E_00*t_end), 0, 0, 0], [0, np.exp(2*np.pi*complex(0, 1)/H*E_01*t_end), 0, 0], [0, 0, np.exp(
        2*np.pi*complex(0, 1)/H*E_10*t_end), 0], [0, 0, 0, np.exp(2*np.pi*complex(0, 1)/H*E_11*t_end)]])
    UG_sub = np.matmul(phase_gate, UG_sub)
    return UG_sub


def dressed_state_index_find(bare_state_list, operator_order_num, dressed_featurevector):
    """The function finding the corresponding dress state's index according to the bare state's tag.

    Args:
        bare_state_list (list[int]): Bare state tag.
        operator_order_num (int): _description_.
        dressed_featurevector (np.array): _description_.

    Returns:
        int: The index of dressed state in dressed_featurevector.
    """
    bare_state_index = 0
    for i in range(3):
        bare_state_index = bare_state_index+bare_state_list[i] * \
            operator_order_num**(2-i)
    return np.argmax(np.abs(dressed_featurevector[bare_state_index, :]))


def Fedelity(UG_sub, matrix):
    """Calculating fedelity.

    Args:
        UG_sub (np.array): Sub time evolution operator.
        matrix (np.array): Target matrix.

    Returns:
        float: Fedelity.
    """
    F = 0
    single_gate = np.zeros([2, 2], dtype=complex)
    single_gate[0][0] = UG_sub[0][0]
    single_gate[0][1] = UG_sub[0][2]
    single_gate[1][0] = UG_sub[2][0]
    single_gate[1][1] = UG_sub[2][2]
    theta_g = (np.angle(single_gate[0][0])+np.angle(single_gate[1][1]))/2.0
    single_gate = single_gate/(np.exp(complex(0, 1)*theta_g))
    identity = np.matmul(single_gate, matrix.transpose().conjugate())
    F = np.abs(identity[0][0]+identity[1][1])/2.0
    return F


def rotation_gate(nx, ny, nz, phi_global, phi):
    """Generating rotation gate's unitary matrix.

    Args:
        nx (float): The X component of the axis of rotation.
        ny (float): The Y component of the axis of rotation.
        nz (float): The Z component of the axis of rotation.
        phi_global (float): Global phase.
        phi (float): Rotation angle.

    Returns:
        np.array: Rotation gate's unitary matrix.
    """
    mode = np.sqrt(nx**2+ny**2+nz**2)
    nx = nx/mode
    ny = ny/mode
    nz = nz/mode
    gate = np.zeros([2, 2], dtype=complex)
    gate[0][0] = np.cos(phi/2)-complex(0, 1)*nz*np.sin(phi/2)
    gate[0][1] = (-complex(0, 1)*nx-ny)*np.sin(phi/2)
    gate[1][0] = (-complex(0, 1)*nx+ny)*np.sin(phi/2)
    gate[1][1] = np.cos(phi/2)+complex(0, 1)*nz*np.sin(phi/2)
    gate = gate*np.exp(complex(0, 1)*phi_global)
    return gate
