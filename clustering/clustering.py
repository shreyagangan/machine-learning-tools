import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import multivariate_normal

################################## FUNCTIONS : KMeans ################################################

def find_centroid(dp,centers):
    print(np.sum(np.square(np.subtract(dp,centers)),axis=1))
    A=np.argmin(np.sum(np.square(np.subtract(dp,centers)),axis=1))
    return A

############################################

def kmeans(kernel,K):
    prototypes=[[]]*K
    #initialize random means
    for i in range(0,K):
        index=random.randint(0,len(kernel))
        print(index)
        prototypes[i]=kernel[index]

    clusters= [[] for _ in range(K)]
    iterations=0
    isSame=False

    while isSame==False & iterations<100:
        newclusters= [[] for _ in range(K)]
        iterations+=1
        i=0
        isSame=True
        for i in range(0,len(kernel)):
            centroid=find_centroid(kernel[i],prototypes)
            print(centroid)
            newclusters[centroid].append(i)

        #check if clusters and new clusters are same
        if(clusters!=newclusters):
            isSame=False
        clusters=newclusters

        #change means
        for i in range(0,K):
            prototypes[i]=np.mean(kernel[newclusters[i]],axis=0)
        print(isSame)

    print(iterations)
    return clusters,prototypes

############################################
def printclusters(clusters,data,K):
    colors=["blue","red","green","purple","yellow"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(0,K):
        ax.scatter(data[clusters[i],[0]],data[clusters[i],[1]],color=colors[i])

#############################################

################################## FUNCTIONS : GMM ################################################
##########################################################################
def emprintclusters(data,K,w,mean,cov):
    colors=["blue","red","green"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for n in range(0,len(data)):
        norm=[0.0]*3
        for k in range(0,3):
            norm[k]=multivariate_normal.pdf(data1[n], mean=mean[k], cov=cov[k])
        #elementwise multiplication
        w_norm=np.multiply(w,norm)
        clust=np.argmax(w_norm)
        ax.scatter(data[n][0],data[n][1],color=colors[clust])
###############################################################
def gmm(data1,w,mean,cov,iters):
    ############### E-Step #####################
    gamma=[[] for _ in range(len(data1))] #nx3 matrix
    logL=[]
    prev=0.0
    for i in range(0,iters):
        print(i)
        for n in range(0,len(data1)):
            norm=[0.0]*3
            for k in range(0,3):
                norm[k]=multivariate_normal.pdf(data1[n], mean=mean[k], cov=cov[k])
            #elementwise multiplication
            w_norm=np.multiply(w,norm)
            px=np.sum(w_norm)
            gamma[n]=np.divide(w_norm,px)

        ### Update parameters ###
        gammak=np.transpose(gamma)
        ### Update w
        for k in range(0,3):
            w[k]=np.divide(np.sum(gammak[k]),600.0)
        if(np.sum(w)<1):
            diff=1-np.sum(w)
            w[0]+=diff

        ### Update mean
        for k in range(0,3):
            xn=np.divide(np.sum(np.multiply(gammak[k],np.transpose(data1)[0])),np.sum(gammak[k]))
            yn=np.divide(np.sum(np.multiply(gammak[k],np.transpose(data1)[1])),np.sum(gammak[k]))
            mean[k]=[xn,yn]

        ### Update covariance
        for k in range(0,3):
            cov[k]=[[0.0,0.0],[0.0,0.0]]
            for n in range(0,len(data1)):
                cov[k]=np.add(cov[k],np.multiply(np.dot(np.subtract(data1[n],mean[k])[np.newaxis].T,np.subtract(data1[n],mean[k])[np.newaxis]),gammak[k][n]))
            #np.multiply(np.dot(np.subtract(data1[1],mean[0])[np.newaxis].T,np.subtract(data1[1],mean[0])[np.newaxis]),gammak[0][1])
            #cov[k]=np.divide(np.dot(np.transpose(np.subtract(data1,mean[k])),np.multiply(np.subtract(data1,mean[k]),np.transpose([gammak[0],]*2))),np.sum(gammak[0]))
            #varx=np.divide(np.sum(np.multiply(gammak[k],np.square(np.subtract(data1_t[0],mean[k][0])))),np.sum(gammak[k]))
            #vary=np.divide(np.sum(np.multiply(gammak[k],np.square(np.subtract(data1_t[1],mean[k][1])))),np.sum(gammak[k]))
            #cov[k]=[[varx,0.0],[0.0,vary]]
            #cov[k]=np.divide(np.sum(np.multiply(np.dot(np.transpose(np.subtract(data1,mean[k]),np.subtract(data1,mean[k])))),axis=0),np.sum(gammak[k]))
            cov[k]=np.divide(cov[k],np.sum(gammak[k]))




        ############### M-Step #####################
        lL=[0.0]*3
        for k in range(0,3):
            for n in range(0,len(data1)):
                norm=multivariate_normal.pdf(data1[n], mean=mean[k], cov=cov[k])
                lL[k]+=gamma[n][k]*(math.log(w[k])+math.log(norm))
        lL=np.sum(lL)
        #Atotal=np.sum(A)
        logL.append(lL)
    return w,mean,cov,logL
########################################################



########################################################### KMeans  ##################################################################################
################################ Q(4).1  ##########################################
#noofclusters=[3]

file1=open("blob.csv")
df1=pd.read_csv(file1,sep=',',names=["x","y"])
file1.close()
data1=df1.as_matrix()

file2=open("circle.csv")
df2=pd.read_csv(file2,sep=',',names=["x","y"])
file2.close()
data2=df2.as_matrix()

################################ Q(4).2  ##########################################
noofclusters=[2,3,5]
for K in noofclusters:
    print('######################### k='+str(K)+' ##################################')
    clust,proto=kmeans(data1,K)
    printclusters(clust,data1,K)
    print(proto)

for K in noofclusters:
    printclusters(kmeans(data2,K)[0],data2,K)

################################ Q(4).3 ###################################################
K=2
newdf=pd.DataFrame(columns=['xsq_plus_ysq'])
newdf['xsq_plus_ysq']=df2['x']**2+df2['y']**2

newdata=newdf.as_matrix()
clust,proto=kmeans(newdata,K)
printclusters(clust,data2,K)

########################################################### GMM  ##################################################################################
data1_t=np.transpose(data1)
colors=["blue","red","green","purple","yellow"]
################ inititalization of parameters
fig1 = plt.figure()
ax2 = fig1.add_subplot(111)

#centroids=[[[] for _ in range(3)] for _ in range(5)]

for runs in range(0,5):
    mean=[[] for _ in range(3)]
    minx=np.min(data1_t[0])
    maxx=np.max(data1_t[0])
    miny=np.min(data1_t[1])
    maxy=np.max(data1_t[1])

    for k in range(0,3):
        mean[k]=[random.uniform(minx, maxx),random.uniform(miny, maxy)]
    cov=[[[np.var(data1_t[0]),0.0],[0.0,np.var(data1_t[1])]]]*3
    w=[0.3333333333333333, 0.3333333333333333, 0.3333333333333333]
    if(np.sum(w)<1):
        diff=1-np.sum(w)
        w[0]+=diff

    w,mean,cov,logL=gmm(data1,w,mean,cov,100)
    print("#############################"+str(runs)+"################################")
    print("weight:")
    print(w)
    print("mean:")
    print(mean)
    print("covariance:")
    print(cov)
    #emprintclusters(data1,3,w,mean,cov)
    ax2.plot(range(1,101),logL,colors[runs])


###################### for best logL ##################################


w=[0.33115823847789605, 0.33550658307949677, 0.33333517844260724]
mean=[[-0.63946289865377592, 1.4746064045257594], [-0.32592106449480068, 0.97133573846690957], [0.7589603247831086, 0.6797698202301814]]
cov=[[[ 0.0359676 ,  0.01549315],
       [ 0.01549315,  0.01935168]],
     [[ 0.03604954,  0.01463887],
       [ 0.01463887,  0.0162912 ]],
     [[ 0.02717056, -0.00840045],
       [-0.00840045,  0.040442  ]]]
w,mean,cov,logL=gmm(data1,w,mean,cov,20)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(1,21),logL)
emprintclusters(data1,3,w,mean,cov)
