from operator import itemgetter
import scipy.stats
import numpy as np
import math

def normalizeTuple(temp,mean,std):
    x=np.subtract(temp,mean)
    x=np.divide(x,std)
    return x

def getClass(id,tuple,training,correct_class):
    correct_count=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    L1={}
    L2={}
    #print(tuple[:9])
    #print(tuple[9])
    ##calculate L1 and L2 from id to all others
    for i in training:
        if not i==id:
            #find absolute difference:
            x=np.abs(np.subtract(tuple,training[i][0:9]))
            L1[i]=sum(x)
            L2[i]=sum(np.square(x))

    ##find the nearest neighbours
    sorted_L1 = sorted(L1.items(), key=itemgetter(1))
    sorted_L2 = sorted(L2.items(), key=itemgetter(1))

    #correct_class=tuple[9]
    #k=1,3,5,7

    #k=1
    index_L1=sorted_L1[0][0]
    class_L1=training[index_L1][9]
    index_L2=sorted_L2[0][0]
    class_L2=training[index_L2][9]

    if(class_L1==correct_class):
        correct_count[0]+=1
    if(class_L2==correct_class):
        correct_count[4]+=1

    class_votes_L1={1:0,2:0,3:0,4:0,5:0,6:0,7:0}
    class_votes_L2={1:0,2:0,3:0,4:0,5:0,6:0,7:0}

    for i in range(0,7):
        index_L1=sorted_L1[i][0] #0 stands for the index
        index_L2=sorted_L2[i][0]
        class_L1=training[index_L1][9]
        class_L2=training[index_L2][9]
        class_votes_L1[int(class_L1)]+=1
        class_votes_L2[int(class_L2)]+=1
        #k=3
        if(i==2):
            k3_L1=max(class_votes_L1.iteritems(), key=itemgetter(1))[0] #k3_L1
            k3_L2=max(class_votes_L2.iteritems(), key=itemgetter(1))[0] #k3_L2
            if(k3_L1==correct_class):
                correct_count[1]+=1
            if(k3_L2==correct_class):
                correct_count[5]+=1
        #k=5
        if(i==4):
            k5_L1=max(class_votes_L1.iteritems(), key=itemgetter(1))[0] #k5_L1
            k5_L2=max(class_votes_L2.iteritems(), key=itemgetter(1))[0] #k5_L2
            if(k5_L1==correct_class):
                correct_count[2]+=1
            if(k5_L2==correct_class):
                correct_count[6]+=1
        #k=7
        if(i==6):
            k7_L1=max(class_votes_L1.iteritems(), key=itemgetter(1))[0] #k7_L1
            k7_L2=max(class_votes_L2.iteritems(), key=itemgetter(1))[0] #k7_L2
            if(k7_L1==correct_class):
                correct_count[3]+=1
            if(k7_L2==correct_class):
                correct_count[7]+=1
    return correct_count


import scipy.stats

def getBayesClass(X,mean_class,SD_class,prior):
    argmax={1:float('-inf'),2:float('-inf'),3:float('-inf'),4:float('-inf'),5:float('-inf'),6:float('-inf'),7:float('-inf'),8:float('-inf'),9:float('-inf')}
    #class by class
    for c in range(1,8):#7 classes
        G=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        for xi in range(0,9): #9 features
            if(SD_class[c][xi]!=0):
                G[xi]=scipy.stats.norm(mean_class[c][xi],SD_class[c][xi]).pdf(X[xi])
        if prior[c]!=0 and np.prod(G)!=0:
            argmax[c]=prior[c]*np.prod(G)
    classmax=max(argmax.iteritems(), key=itemgetter(1))[0]
    return classmax

