import numpy as np


def cos_matrix_n(matrix, n):
    """The function of cosine's taylor expansion. The independent variable is a matrix and 
    matrix multiplication is not scalar multiplication.

    Args:
        matrix (np.array): The independent variable matrix to be expanded.
        n (int): Order of expansion.

    Returns:
        np.array: return matrix.
    """
    result = 0
    n = int((n-n % 2)/2+1)
    for k in range(n):
        result = result + \
            np.power(-1, k)*np.linalg.matrix_power(matrix, 2*k) / \
            np.math.factorial(2*k)
    return result


def sin_matrix_n(matrix, n):
    """The function of sine's taylor expansion. The independent variable is a matrix and 
    matrix multiplication is not scalar multiplication.

    Args:
        matrix (np.array): The independent variable matrix to be expanded.
        n (int): Order of expansion.

    Returns:
        np.array: return matrix.
    """
    result = 0
    n = int((n+n % 2)/2)
    for k in range(n):
        result = result + \
            np.power(-1, k) * np.linalg.matrix_power(matrix, 2*k+1) / \
            np.math.factorial(2*k+1)
    return result


def cos_alpha_matrix_n(alpha, matrix, n):
    """Taylor expansion of cos(alpha+matrix). The independent variable is a matrix and 
    matrix multiplication is not scalar multiplication.

    Args:
        alpha (float): Phase.
        matrix (np.array): The independent variable matrix to be expanded.
        n (int): Order of expansion.

    Returns:
        np.array: return matrix.
    """
    result = np.cos(alpha)*cos_matrix_n(matrix, n) - \
        np.sin(alpha)*sin_matrix_n(matrix, n)
    return result


def exp_matrix_n(matrix, n):
    """The function of exponent's taylor expansion. The independent variable is a matrix and 
    matrix multiplication is not scalar multiplication.

    Args:
        matrix (np.array): The independent variable matrix to be expanded.
        n (int): Order of expansion.

    Returns:
        np.array: return matrix.
    """
    result = 0
    n = n+1
    for k in range(n):
        result = result + \
            np.linalg.matrix_power(matrix, k)/np.math.factorial(k)
    return result


def annihilation_operator_n(n):
    """The function generating annihilation operator of order n.

    Args:
        n (int): The order of matrix.

    Returns:
        np.array: return matrix.
    """
    result = np.zeros([n, n])
    for i in range(n-1):
        result[i][i+1] = np.sqrt(i+1)
    return result


def creation_operator_n(n):
    """The function generating creation operator of order n.

    Args:
        n (int): The order of matrix.

    Returns:
        np.array: return matrix.
    """
    result = np.zeros([n, n])
    for i in range(n-1):
        result[i+1][i] = np.sqrt(i+1)
    return result
