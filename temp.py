import function as fun
import numpy as np

Y=complex(0,1)*(fun.annihilation_operator_n(8)-fun.creation_operator_n(8))/np.sqrt(2)
print(np.linalg.matrix_power(Y,2))
