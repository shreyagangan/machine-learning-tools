import numpy as np
from functions import getClass
from functions import normalizeTuple
from functions import getBayesClass

##normalizing the data

#read training data and get sum in mean
training={}
mean=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

mean_class={1:[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
            2:[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
            3:[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
            4:[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
            5:[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
            6:[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
            7:[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]}

SD_class={1:[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
          2:[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
          3:[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
          4:[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
          5:[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
          6:[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
          7:[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]}

prior={1:0.0,
       2:0.0,
       3:0.0,
       4:0.0,
       5:0.0,
       6:0.0,
       7:0.0}



#find std deviation
std=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

with open("train.txt","r") as f:
    for line in f:
        splitLine = map(float,line.strip().split(","))
        training[int(splitLine[0])] = splitLine[1:]
        mean=np.add(mean,splitLine[1:10])
        sample_class=int(splitLine[10])
        mean_class[sample_class]=np.add(mean_class[sample_class],splitLine[1:10])
        prior[sample_class]+=1

#print(prior)
N=len(training)
#find mean
mean=np.divide(mean, len(training))

for i in range(1,8):
    if not prior[i]==0:
        mean_class[i]=np.divide(mean_class[i],prior[i])

for i in training:
    x=np.square(np.abs(np.subtract(training[i][0:9],mean)))
    std=np.add(std,x)
    sample_class=int(training[i][9])
    temp=np.square(np.subtract(training[i][0:9],mean_class[sample_class]))
    SD_class[sample_class]=np.add(SD_class[sample_class],temp)

for i in range(1,8):
    if not prior[i]==0:
        SD_class[i]=np.sqrt(np.divide(SD_class[i],prior[i]))
    prior[i]=prior[i]/N


std=np.divide(std,len(training)-1)
std=np.sqrt(std)

for i in range(0,9):
    if(std[i]==0):
        std[i]=1
        mean[i]=0

#For Training
accuracy=0.0
for id in training:
    true_class=training[id][9]
    class_label=getBayesClass(training[id][0:9],mean_class,SD_class,prior)
    if int(true_class)==int(class_label):
        accuracy+=1

print("Results for Naive Bayes:")

print("Accuracy for Training Data:")
print(accuracy/N)


#For Test Data
accuracy2=0.0
test_count2=0
with open("test.txt","r") as f:
    for line in f:
        test_count2+=1
        splitLine = map(float,line.strip().split(","))
        tuple=splitLine[1:10]
        true_class2=splitLine[10]
        class_label2=getBayesClass(tuple,mean_class,SD_class,prior)
        if int(true_class2)==int(class_label2):
            accuracy2+=1
print("Accuracy for Test Data:")
print(accuracy2/test_count2)


for i in training:
    temp=training[i][0:9]
    x=normalizeTuple(temp,mean,std)
    training[i][0:9]=x

#id,[k1_L1,k3_L1,k5_L1,k7_L1,k1_L2,k3_L2,k5_L2,k7_L2]
accuracy=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

for id in training:
    temp=getClass(id,training[id][:9],training,training[id][9])
    accuracy=np.add(accuracy,temp)

print("Results for KNN:")

print("Accuracy for Training Data: [(k=1,L1),(k=3,L1),(k=5,L1),(k=7,L1),(k=1,L2),(k=3,L2),(k=5,L2),(k=7,L2)] ")
print(np.divide(accuracy,len(training)))
#print(accuracy)

accuracy2=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
test_count=0
with open("test.txt","r") as f:
    for line in f:
        test_count+=1
        splitLine = map(float,line.strip().split(","))
        id=int(splitLine[0])
        tuple=splitLine[1:10]
        true_class=splitLine[10]
        tuple=normalizeTuple(tuple,mean,std)
        temp=getClass(id,tuple,training,true_class)
        #print(temp)
        accuracy2=np.add(accuracy2,temp)

print("Accuracy for Testing Data: [(k=1,L1),(k=3,L1),(k=5,L1),(k=7,L1),(k=1,L2),(k=3,L2),(k=5,L2),(k=7,L2)] ")
print(np.divide(accuracy2,test_count))


