import numpy as np
import SFQControl.quantum as sq
from scipy import special

a=sq.rotation_gate(1,0,0,0,np.pi/2)
print(a.shape[0])

