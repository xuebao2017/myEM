# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import math
import copy

def getData(MU, SIGMA, N):
    global mu
    global sigma
    global expection
    global orimu
    global orisigma
    # mu=np.array([10.0,60.0])
    # sigma=np.array([8.0,3.0])
    mu=np.random.random(2)*10
    sigma=np.random.random(2)*10
    orimu=copy.deepcopy(mu)
    orisigma=copy.deepcopy(sigma)
    expection=np.zeros((N,2))
    data=[]
    for i in xrange(N):
        flag=np.random.random()
        if flag>0.5:
            data.append(np.random.normal(MU[1], SIGMA[1]))
        else:
            data.append(np.random.normal(MU[0], SIGMA[0]))
    return data

def showData(data):
    plt.hist(data,50)
    plt.show()

# 求期望，求得每个样本点属于每个正态分布的概率
def e_step(data):
    global mu
    global expection
    global sigma
    N=len(data)
    for i in xrange(N):
        probility=[0.0]*2
        total=0
        for j in xrange(2):
            probility[j]=(1/(math.sqrt(2*math.pi)*sigma[j]))*math.exp(-1*(((data[i]-mu[j])**2)/(2*(sigma[j]**2))))
            total=total+probility[j]
        for j in xrange(2):
            expection[i,j]=probility[j]/total

def m_step(data):
    global mu
    global sigma
    global expection
    N=len(data)
    #保存上一步的mu,求sigma时需要用到
    last_mu=copy.deepcopy(mu)
    #估计参数mu
    for j in xrange(2):
        fenzi = 0
        fenmu = 0
        for i in xrange(N):
            fenzi+=expection[i,j]*data[i]
            fenmu+=expection[i,j]
        mu[j]=fenzi/fenmu
    #估计参数sigma
    for j in xrange(2):
        fenzi=0
        fenmu=0
        for i in xrange(N):
            fenzi+=expection[i,j]*((data[i]-last_mu[j])**2)
            fenmu+=expection[i,j]
        sigma[j]=math.sqrt(fenzi/fenmu)

def runEm_gmm():
    cap=0.001
    MU=[20,50]
    SIGMA=[10,5]
    N=1000
    data = getData(MU, SIGMA, N)
    max_iters=1000
    for i in xrange(max_iters):
        last_mu=copy.deepcopy(mu)
        last_sigma=copy.deepcopy(sigma)
        e_step(data)
        m_step(data)
        print "iter %d times"%i
        if sum(abs(last_mu-mu))<cap:
            if sum(abs(last_sigma-sigma)<cap):
                break
    print "正确的mu为 ",MU
    print "正确的sigma为 ",SIGMA
    print "------------------------"
    print "初始的mu为 ",orimu
    print "初始的sigma为 ",orisigma
    print "------------------------"
    print "估计的mu为 ",mu
    print "估计的sigma为 ",sigma
    showData(data)

runEm_gmm()