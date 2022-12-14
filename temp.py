import numpy as np

mat=np.random.rand(27,27)
mat=mat+np.random.rand(27,27)*complex(0,1)
eigenvalue, featurevector = np.linalg.eig(mat)



print("特征值：", eigenvalue)
print("特征向量：", featurevector)