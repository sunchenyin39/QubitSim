import numpy as np


def Gaussian_function_list_generator(x_list, a, b, c):
    """a*exp(-(x-b)**2/c**2/2).

    Args:
        x_list (list[float]): Independent variable list.
        a (float): a*exp(-(x-b)**2/c**2/2)
        b (float): a*exp(-(x-b)**2/c**2/2)
        c (float): a*exp(-(x-b)**2/c**2/2)

    Returns:
        list[float]: Gaussian function list.
    """
    Gaussian_function_list = a*np.exp(-(x_list-b)*(x_list-b)/2/c**2)
    return Gaussian_function_list


def Gaussian_function_sequence_generator(x_list, a, b_list, c):
    """a*exp(-(x-b)**2/c**2/2).

    Args:
        x_list (list[float]): Independent variable list.
        a (float): a*exp(-(x-b)**2/c**2/2)
        b_list (list[float]): a*exp(-(x-b)**2/c**2/2)
        c (float): a*exp(-(x-b)**2/c**2/2)

    Returns:
        list[float]: Gaussian function sequence list.
    """
    Gaussian_function_sequence_list = np.zeros(len(x_list))
    for i in range(len(x_list)):
        for j in range(len(b_list)):
            Gaussian_function_sequence_list[i] = Gaussian_function_sequence_list[i]+a*np.exp(-(
                x_list[i]-b_list[j])*(x_list[i]-b_list[j])/2/c**2)
    return Gaussian_function_sequence_list
