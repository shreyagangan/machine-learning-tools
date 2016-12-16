import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#############################################################

#Generate x,y
def dataset_generation(size):
    x = np.random.uniform(-1,1,size)
    e = np.random.normal(0,0.1,size)
    sq=np.multiply(x,x)
    sq2=np.multiply(sq,2)
    y = np.add(sq2,e)
    return x,y

#Create a feature set given x and the number of dependent variable terms
#for example: g3(x)=w0+w1x+w2xsq -> feature_set(x,2)
def feature_set(x,gindex):
    no=gindex-2
    x_set=pd.DataFrame(x,columns=["1"])
    for i in range(2,no+1):
        x_set[str(i)]=x_set[str(i-1)]*x
    x_set.insert(0,"0",1)
    if(no==0):
        return x_set[["0"]]
    return x_set

#linear regression theta
def linear_regression(X,Y):
    #X.T.dot(X)
    a=np.linalg.pinv(pd.DataFrame.as_matrix(X.T.dot(X)))
    b=a.dot(X.T)
    theta=b.dot(Y)
    return theta

#Ridge Regression
#theta
def ridge_regression(X,Y,rlambda):
    n=len(X.columns)
    I=np.identity(n)
    I[0][0]=0.0
    I=len(Y)*rlambda*I
    theta=np.dot(np.dot(np.linalg.pinv(pd.DataFrame.as_matrix(X.T.dot(X))+I),np.asarray(X.T)),np.asarray(Y))
    return theta


def predict(X,theta):
    prediction=X.dot(theta) #theta.dot(f.T)
    return prediction

def get_SSE(prediction,Y):
    diff=np.subtract(Y,prediction)
    sdiff=diff**2
    SSE=np.sum(sdiff)
    return SSE

#################################################################################
def linreg_bias_variance(D,size,data_x,data_y):
    #for g=1 remaining
    result1="g1"
    MSE1=[]
    var_pred1=[]
    for i in range(0,D):
        prediction=[1]*size
        SSE=get_SSE(prediction,data_y[i])
        var_pred1.append(np.var(prediction))
        MSE1.append(SSE/size)

    #plot histogram of MSE
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(MSE1,bins=10)

    #VARIANCE
    variance1=np.mean(var_pred1)
    result1=result1+"\t "+str(variance1)
    #BIAS2
    result1=result1+"\t\t "+str(np.mean(MSE1))
    print(result1)

    ####LINEAR REGRESSION
    for g in range(2,7):    #for each g
        result="g"+str(g)

        MSE=[]
        var_pred=[]
        for i in range(0,D):
            f=feature_set(data_x[i],g)
            theta=linear_regression(f,data_y[i])
            prediction=predict(f,theta)
            var_pred.append(np.var(prediction))
            SSE=get_SSE(prediction,data_y[i])
            MSE.append(SSE/size)

        #plot histogram of MSE
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(MSE,bins=10)

        #VARIANCE
        variance=np.mean(var_pred)
        result=result+"\t "+str(variance)

        #BIAS2
        result=result+"\t\t "+str(np.mean(MSE))
        print(result)

######################################################################

def ridgereg_bias_variance(D,size,data_x,data_y,lam):
    g=4
    result=str(lam)

    MSE=[]
    var_pred=[]
    for i in range(0,D):
        f=feature_set(data_x[i],g)
        theta=ridge_regression(f,data_y[i],lam)
        prediction=predict(f,theta)
        var_pred.append(np.var(prediction))
        SSE=get_SSE(prediction,data_y[i])
        MSE.append(SSE/size)

    #VARIANCE
    variance=np.mean(var_pred)
    result=result+"\t "+str(variance)

    #BIAS2
    result=result+"\t\t "+str(np.mean(MSE))
    print(result)


########################### Q.5 (a)  ##################################

################# DATA GENERATION ###################
data_x=[]
data_y=[]
D=100
size=10

for i in range(0,D):
    x,y=dataset_generation(size)
    data_x.append(x)
    data_y.append(y)
print("################ Q.5(a) LINEAR REGRESSION ######################")
print("For 100 DataSets of size 10:")
print("g \t VARIANCE\t\t BIAS^2")
linreg_bias_variance(D,size,data_x,data_y)


########################### Q.5 (b)  ##################################

################# DATA GENERATION ###################
data_x=[]
data_y=[]
D=100
size=100

for i in range(0,D):
    x,y=dataset_generation(size)
    data_x.append(x)
    data_y.append(y)

print("################ Q.5(b) LINEAR REGRESSION ######################")
print("For 100 DataSets of size 100:")
print("g \t VARIANCE\t\t BIAS^2")
linreg_bias_variance(D,size,data_x,data_y)


########################### Q.5 (d)  ##################################

lmda=[0.001,0.003,0.01,0.03,0.1,0.3,1.0]
print("################ Q.5(d) RIDGE REGRESSION ######################")
print("For 100 DataSets of size 100:")
print("lambda \t VARIANCE\t\t BIAS^2")
for l in lmda:
    ridgereg_bias_variance(D,size,data_x,data_y,l)