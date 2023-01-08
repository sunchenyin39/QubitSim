import numpy as np
import SFQControl.quantum
import SFQControl.GA


def main():
    parameters=SFQControl.GA.Parameters()
    parameters.Nc = 300  # subsequence所包含的SFQ时钟周期数
    parameters.omegaSFQ = 2*np.pi*25E9  # SFQ时钟频率
    parameters.times = 1  # subsequence重复的次数
    parameters.T_SFQclock = 2*np.pi/parameters.omegaSFQ  # SFQ时钟周期
    parameters.USFQ = np.load("X2PQ1_idleQ2_SFQ_1_matrix.npy")
    parameters.UFR = np.load("X2PQ1_idleQ2_SFQ_0_matrix.npy")
    parameters.dressed_eigenvalue=np.load("X2PQ1_idleQ2_dressed_eigenvalue.npy")
    parameters.dressed_featurevector=np.load("X2PQ1_idleQ2_dressed_featurevector.npy")
    parameters.operator_order_num=4
    parameters.popsize = 100  # GA算法的种群数量
    parameters.itenumber =1000  # GA算法的繁衍次数
    parameters.power = 40  # GA算法的复制函数参数 power<popsiza/2
    parameters.pc = 0.9  # GA算法的交叉概率
    parameters.pm = 1  # GA算法的变异概率
    parameters.mutnumber = 100  # 单次变异基因数量
    parameters.targetfedelity = 0.9999  # GA算法的目标保真度
    parameters.matrix = SFQControl.quantum.rotation_gate(1,0,0,0,np.pi/2)  # 目标单比特门
    parameters.popfilename = 'pop.npy'  # 初始种群文件，若打不开则随机生成
    SFQControl.GA.GA(parameters)


if __name__ == '__main__':
    main()
