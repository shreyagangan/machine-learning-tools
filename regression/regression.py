import itertools
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
####################################################################################
def histogram(train_data):
    for i in range(1,15):
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.hist(train_data[str(i)],bins=10)

def pearson_coeff(X,Y):
    N=len(Y)
    numr=(N*(X.T.dot(Y)))-(np.sum(X)*np.sum(Y))
    sumX=np.sum(X)
    A=(N*np.sum(np.square(X)))-(sumX*sumX)
    sumY=np.sum(Y)
    B=(N*np.sum(np.square(Y)))-(sumY*sumY)
    denr=A*B
    p=numr/math.sqrt(denr)
    #print(p)
    return p

#Linear Regression
#theta
def theta_linear(X,Y):
    theta=np.dot(np.dot(np.linalg.pinv(np.asarray(X.T.dot(X))),np.asarray(X.T)),np.asarray(Y))
    return theta

def mse_linear(X,Y,theta):
    Y_pred=np.dot(np.asarray(X),theta)
    diff=np.asarray(Y)-Y_pred
    mse=np.sum(diff**2)
    mse=mse/len(Y)
    #print mse
    return mse

def residue(X,Y,theta):
    Y_pred=np.dot(np.asarray(X),theta)
    res=np.asarray(Y)-Y_pred
    #print mse
    return res

#LR with 4 features
def linear_four(combs,X,Y):
    mse=float("inf")
    theta=[]
    combo_no=()
    for combo in combs:
        Xcom=X[['0',combo[0],combo[1],combo[2],combo[3]]]
        theta_com=theta_linear(Xcom,Y)
        mse_com=mse_linear(Xcom,Y,theta_com)
        if(mse_com<mse):
            mse=mse_com
            theta=theta_com
            combo_no=combo

    return theta,combo_no

#Ridge Regression
#theta
def theta_ridge(X,Y,rlambda):
    n=len(X.columns)
    I=np.identity(n)
    I[0][0]=0.0
    I=len(Y)*rlambda*I
    theta=np.dot(np.dot(np.linalg.pinv(np.asarray(X.T.dot(X))+I),np.asarray(X.T)),np.asarray(Y))
    return theta

#Ridge Regression
def ridge(X,Y,theta):
    #rlambda=0.01
    Y_pred_r=np.dot(np.asarray(X),theta)
    diff_r=np.asarray(Y)-Y_pred_r
    mse_r=np.sum(diff_r**2)
    mse_r=mse_r/len(Y)
    #print mse_r
    return mse_r

###########################################
#for given lambda calculate 10-fold CV mse
def mse_cv(X,Y,l):
    mse_sum=0
    counter=0
    #1st 3 iterations will have 44 samples
    for i in range(1,4):
        X_train=pd.concat([X.loc[:counter-1],X.loc[counter+44:]])
        Y_train=pd.concat([Y.loc[:counter-1],Y.loc[counter+44:]])
        X_test=X.loc[counter:counter+43]
        Y_test=Y.loc[counter:counter+43]
        t=theta_ridge(X_train,Y_train,l)
        m=ridge(X_test,Y_test,t)
        mse_sum+=m
        counter+=44
    #Next 7 iterations will have 43 samples each
    for i in range(4,11):
        X_train=pd.concat([X.loc[:counter-1],X.loc[counter+43:]])
        Y_train=pd.concat([Y.loc[:counter-1],Y.loc[counter+43:]])
        X_test=X[counter:counter+42]
        Y_test=Y[counter:counter+42]
        t=theta_ridge(X_train,Y_train,l)
        m=ridge(X_test,Y_test,t)
        mse_sum+=m
        counter+=43
    mse_avg=mse_sum/10.0
    return mse_avg

#find optimal lambda which has least avg mse
def lambda_opt(X,Y):
    min_lambda=-1
    mse_min=float("inf")
    l=0.0001
    while(l<=10):
        mse_lambda=mse_cv(X,Y,l)
        if(mse_lambda<mse_min):
            mse_min=mse_lambda
            min_lambda=l
        l+=0.1
    return min_lambda,mse_min


####################################################################################

#Data Input

#read file into dict
def file_to_dict(filename):
    data=[]
    count=0
    from itertools import izip
    with open(filename,"r") as f:
        for line in f:
            count+=1
            #print(line)
            splitLine = map(float,line.strip().split(","))
            index=['1','2','3','4','5','6','7','8','9','10','11','12','13','14']
            i = iter(splitLine)
            b=dict(izip(index, i))
            data.append(b)
    return data

#find mean and std
def get_mean(X):
    n=len(X.loc[0])
    mean=[0]*n
    for i in range(1,n+1):
        attribute_i=X[str(i)]
        mean[i-1]=np.mean(attribute_i)
    return mean

def get_std(X):
    n=len(X.loc[0])
    std=[0]*n
    #std=[0,0,0,0,0,0,0,0,0,0,0,0,0]
    for i in range(1,n+1):
        attribute_i=X[str(i)]
        std[i-1]=np.std(attribute_i,ddof=1)
    return std

#train_data[['1','2','3','4','5','6','7','8','9','10','11','12','13']]
def normalize(X,mean,std):
    X_norm=np.divide(np.asarray(X)- mean,std)
    return X_norm

##########################################################################################\

#Correlation

#Find 4 features with highest correlation (f1,f2,f3,f4)

########################################### TRAINING DATA PRE-PROCESS #########################################

#read file into dict
data=file_to_dict("housing-data-training.txt")

#dict to dataframe
train_data=pd.DataFrame(data)

#histogram
histogram(train_data)

X=train_data[['1','2','3','4','5','6','7','8','9','10','11','12','13']]
Y=train_data['14']
#print(X)
#print(Y)

#pearson coefficient
pc=[]
for i in range(1,14):
    pc.append(pearson_coeff(X[str(i)],Y))
pearson_c=pd.Series(pc,index=['1','2','3','4','5','6','7','8','9','10','11','12','13'])

#print(pc)
#print(pearson_c)

#find mean and std for each attribute
mean=get_mean(X)
std=get_std(X)

#print("Normalize:")
Xn=pd.DataFrame(normalize(X,mean,std),columns=['1','2','3','4','5','6','7','8','9','10','11','12','13'])
Xn.insert(0, '0', 1)

######################################## TEST DATA ###########################################

#read file into dict
data_test=file_to_dict("housing-data-test.txt")

#dict to dataframe
test_data=pd.DataFrame(data_test)

Xt=test_data[['1','2','3','4','5','6','7','8','9','10','11','12','13']]
Yt=test_data['14']
#print(Xt)
#print(Yt)

#print("Normalize:")
Xtn=pd.DataFrame(normalize(Xt,mean,std),columns=['1','2','3','4','5','6','7','8','9','10','11','12','13'])
Xtn.insert(0, '0', 1)


#################################################### Linear Regression #################################################################

print("3.2 Linear Regression:")
######TRAINING
#parameters
theta=theta_linear(Xn,Y)
#print("Optimal parameters:"+str(theta))
#MSE
mse_lin=mse_linear(Xn,Y,theta)
######TEST

#MSE
mse_lin_t=mse_linear(Xtn,Yt,theta)
print("MSE training data="+str(mse_lin))
print("MSE test data="+str(mse_lin_t))
#################################################### Ridge Regression #################################################################

######TRAINING
print("#####################################################")
print("3.2 Ridge Regression:")
#parameters
t1=theta_ridge(Xn,Y,0.01)
t2=theta_ridge(Xn,Y,0.1)
t3=theta_ridge(Xn,Y,1.0)

#MSE
mse_t1=ridge(Xn,Y,t1)
mse_t2=ridge(Xn,Y,t2)
mse_t3=ridge(Xn,Y,t3)

######TEST
mse_t1_t=ridge(Xtn,Yt,t1)
mse_t2_t=ridge(Xtn,Yt,t2)
mse_t3_t=ridge(Xtn,Yt,t3)


print("For lambda="+str(0.01))
#print("Optimal parameters:"+str(t1))
print("MSE training data="+str(mse_t1))
print("MSE test data="+str(mse_t1_t))

print("For lambda="+str(0.1))
#print("Optimal parameters:"+str(t2))
print("MSE training data="+str(mse_t2))
print("MSE test data="+str(mse_t2_t))

print("For lambda="+str(1.0))
#print("Optimal parameters:"+str(t2))
print("MSE training data="+str(mse_t3))
print("MSE test data="+str(mse_t3_t))

######################################################################

print("#####################################################")
print("3.2 Ridge Regression with Cross-Validation (lambda, mse for test data):")
ans_cv=lambda_opt(Xn,Y)
print(ans_cv)
print("lambda=0.01, mse(for training)="+str(mse_cv(Xn,Y,0.001)))
print("lambda=0.1, mse(for training)="+str(mse_cv(Xn,Y,0.1)))
print("lambda=0.2, mse(for training)="+str(mse_cv(Xn,Y,0.2)))
print("lambda=0.5, mse(for training)="+str(mse_cv(Xn,Y,0.5)))
print("lambda=0.8, mse(for training)="+str(mse_cv(Xn,Y,0.8)))
print("lambda=1.0, mse(for training)="+str(mse_cv(Xn,Y,1.0)))
print("lambda=5.0, mse(for training)="+str(mse_cv(Xn,Y,5.0)))




################################################## Feature Selection #####################################################################

pc=np.abs(pc)

######Correlation - 4 features with highest correlation

print("##########################################################")
print("3.3 Selection with Correlation - (a): 4 features with highest correlation")

#Find 4 features with highest correlation (f1,f2,f3,f4)
f1=np.argmax(pc) #do +1
pc_1=list(pc[:f1])+[-5]
if((f1+1)<len(pc)):
     pc_1+=pc[(f1+1):]
f2=np.argmax(pc_1)
pc_2=pc_1[:f2]+[-5]
if((f2+1)<len(pc)):
     pc_2+=pc_1[(f2+1):]
f3=np.argmax(pc_2)
pc_3=pc_2[:f3]+[-5]
if((f3+1)<len(pc)):
     pc_3+=pc_2[(f3+1):]
f4=np.argmax(pc_3)

attributes_set=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]

print("Attributes with highest correlation with target:")
print(attributes_set[f1])
print(attributes_set[f2])
print(attributes_set[f3])
print(attributes_set[f4])

######TRAINING
#Training Data
Xc=Xn[[str('0'),str(f1+1),str(f2+1),str(f3+1),str(f4+1)]]

#parameters
theta_c=theta_linear(Xc,Y)
#MSE

print("MSE training data="+str(mse_linear(Xc,Y,theta_c)))

######TEST
#Data
Xtc=Xtn[[str('0'),str(f1+1),str(f2+1),str(f3+1),str(f4+1)]]
#MSE
print("MSE test data="+str(mse_linear(Xtc,Yt,theta_c)))


###### Iterative Selection ########
print("##########################################################")
print("3.3 Selection with Correlation - (b): Iterative Selection")

feature_no=['1','2','3','4','5','6','7','8','9','10','11','12','13']

farray=['0']
f=str(np.argmax(pc)+1) #do +1
farray.append(f)
feature_no.remove(f)
theta_i=[]
for j in range(0,4):
    theta_i=theta_linear(Xn[farray],Y)
    r=residue(Xn[farray],Y,theta_i)
    pmax=-float("inf")
    maxindex=-1
    for i in range(0,len(feature_no)):
        feature=feature_no[i]
        pcoeff=abs(pearson_coeff(Xn[feature],r))
        #print pcoeff
        if(pmax<pcoeff):
            pmax=pcoeff
            maxindex=i
    f=str(feature_no[maxindex])
    farray.append(f)
    feature_no.remove(f)
farray.remove(f)#5th feature
#print theta_i
print("Selected Features:["+attributes_set[int(farray[1])-1]+", "+attributes_set[int(farray[2])-1]+", "+attributes_set[int(farray[3])-1]+", "+attributes_set[int(farray[4])-1]+"]")
print("MSE Training Data="+str(mse_linear(Xn[farray],Y,theta_i)))
print("MSE Testing Data="+str(mse_linear(Xtn[farray],Yt,theta_i)))


###### Brute Force ########
print("##########################################################")

print("3.3 Selection with Brute-Force Search:")
a=['1','2','3','4','5','6','7','8','9','10','11','12','13']
combs=list(itertools.combinations(a, 4))

brute=linear_four(combs,Xn,Y)
theta_brute=brute[0]
combo=brute[1]
mse_brute=mse_linear(Xn[['0',combo[0],combo[1],combo[2],combo[3]]],Y,theta_brute)
print("Selected Features:["+attributes_set[int(combo[0])-1]+", "+attributes_set[int(combo[1])-1]+", "+attributes_set[int(combo[2])-1]+", "+attributes_set[int(combo[3])-1]+"]")
print("MSE Training Data="+str(mse_brute))
mse_brute_t=mse_linear(Xtn[['0',combo[0],combo[1],combo[2],combo[3]]],Yt,theta_brute)
print("MSE Test Data="+str(mse_brute_t))


###### Polynomial Feature Expansion ########
print("##########################################################")
print("3.4 Polynomial Feature Expansion:")
poly_combs=list(itertools.combinations_with_replacement(a, 2))



###### INPUT #####
#training
counter=len(X.loc[0])
for combo in poly_combs:
    c=str(counter)
    i=str(combo[0])
    j=str(combo[1])
    X.loc[:,(c)]=np.multiply(np.asarray(X[i]),np.asarray(X[j]))
    #X[c]=np.multiply(np.asarray(X[i]),np.asarray(X[j]))
    counter+=1

#find mean and std for each attribute
meanp=get_mean(X)
stdp=get_std(X)

#print("Normalize:")
Xpn=pd.DataFrame(normalize(X,meanp,stdp),columns=['1','2','3','4','5','6','7','8','9','10',
                                                 '11','12','13','14','15','16','17','18','19','20',
                                                 '21','22','23','24','25','26','27','28','29','30',
                                                 '31','32','33','34','35','36','37','38','39','40',
                                                 '41','42','43','44','45','46','47','48','49','50',
                                                 '51','52','53','54','55','56','57','58','59','60',
                                                 '61','62','63','64','65','66','67','68','69','70',
                                                 '71','72','73','74','75','76','77','78','79','80',
                                                 '81','82','83','84','85','86','87','88','89','90',
                                                 '91','92','93','94','95','96','97','98','99','100',
                                                 '101','102','103'])
Xpn.insert(0, '0', 1)


#testing
counter=len(Xt.loc[0])
for combo in poly_combs:
    c=str(counter)
    i=str(combo[0])
    j=str(combo[1])
    Xt.loc[:,(c)]=np.multiply(np.asarray(Xt[i]),np.asarray(Xt[j]))
    #X[c]=np.multiply(np.asarray(X[i]),np.asarray(X[j]))
    counter+=1

#print("Normalize:")
Xptn=pd.DataFrame(normalize(Xt,meanp,stdp),columns=['1','2','3','4','5','6','7','8','9','10',
                                                 '11','12','13','14','15','16','17','18','19','20',
                                                 '21','22','23','24','25','26','27','28','29','30',
                                                 '31','32','33','34','35','36','37','38','39','40',
                                                 '41','42','43','44','45','46','47','48','49','50',
                                                 '51','52','53','54','55','56','57','58','59','60',
                                                 '61','62','63','64','65','66','67','68','69','70',
                                                 '71','72','73','74','75','76','77','78','79','80',
                                                 '81','82','83','84','85','86','87','88','89','90',
                                                 '91','92','93','94','95','96','97','98','99','100',
                                                 '101','102','103'])
Xptn.insert(0, '0', 1)


##### LR ####
######TRAINING
#parameters
thetap=theta_linear(Xpn,Y)
#print("Optimal parameters:"+str(theta))
#MSE
mse_p=mse_linear(Xpn,Y,thetap)
print("MSE training data="+str(mse_p))

######TEST
#MSE
mse_p_t=mse_linear(Xptn,Yt,thetap)
print("MSE test data="+str(mse_p_t))
