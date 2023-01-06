import numpy as np


def Gaussian_function_list_generator(x_list, a, b, c):
    """a*exp(-(x-b)**2/c**2/2)

    Args:
        x_list (list[float]): Independent variable list.
        a (_type_): a*exp(-(x-b)**2/c**2/2)
        b (_type_): a*exp(-(x-b)**2/c**2/2)
        c (_type_): a*exp(-(x-b)**2/c**2/2)

    Returns:
        _type_: _description_
    """    
    Gaussian_function_list = a*np.exp(-(x_list-b)*(x_list-b)/2/c**2)
    return Gaussian_function_list
