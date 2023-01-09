import numpy as np
import SFQControl.quantum
import SFQControl.GA
import copy


def main():
    Nc = 500  # subsequence所包含的SFQ时钟周期数
    omegaSFQ = 2*np.pi*25E9  # SFQ时钟频率
    times = 1  # subsequence重复的次数
    T_SFQclock = 2*np.pi/omegaSFQ  # SFQ时钟周期
    USFQ = np.load("X2PQ1_idleQ2_SFQ_1_matrix.npy")
    UFR = np.load("X2PQ1_idleQ2_SFQ_0_matrix.npy")
    dressed_eigenvalue = np.load("X2PQ1_idleQ2_dressed_eigenvalue.npy")
    dressed_featurevector = np.load("X2PQ1_idleQ2_dressed_featurevector.npy")
    operator_order_num = 4
    popsize = 100  # GA算法的种群数量
    itenumber = 50  # GA算法的繁衍次数
    power = 40  # GA算法的复制函数参数 power<popsiza/2
    pc = 0.9  # GA算法的交叉概率
    pm = 0.9  # GA算法的变异概率
    mutnumber = 100  # 单次变异基因数量
    targetfedelity = 0.9995  # GA算法的目标保真度
    matrix = SFQControl.quantum.rotation_gate(1, 0, 0, 0, np.pi/2)  # 目标单比特门
    popfilename = 'pop.npy'  # 初始种群文件，若打不开则随机生成

    pop = np.load(popfilename)
    popFedelity = SFQControl.GA.popFedelity(
        pop, USFQ, UFR, times, matrix, operator_order_num, dressed_eigenvalue, dressed_featurevector, T_SFQclock)
    individual_fedelity=np.max(popFedelity)
    individual=pop[np.argmax(popFedelity)]
    np.save("individual.npy",individual)
    # individual = np.load("individual.npy")
    # individual_fedelity = SFQControl.quantum.Fedelity(SFQControl.quantum.UGgenerator(USFQ, UFR, individual, times, operator_order_num, dressed_eigenvalue, dressed_featurevector, T_SFQclock), matrix)
    # print(individual_fedelity)
    # for i in range(len(individual)):
    #     individual_new = copy.deepcopy(individual)
    #     individual_new[i]=(individual_new[i]+1)%2
    #     individual_new_fedelity=SFQControl.quantum.Fedelity(SFQControl.quantum.UGgenerator(USFQ, UFR, individual_new, times, operator_order_num, dressed_eigenvalue, dressed_featurevector, T_SFQclock), matrix)
    #     if individual_new_fedelity>individual_fedelity:
    #         np.save("individual.npy",individual_new)
    #         break

if __name__ == '__main__':
    main()
