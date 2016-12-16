import sys
import time
sys.path.append('libsvm-3.21/python')
from svmutil import *


y, x = svm_read_problem('training.txt')
prob = svm_problem(y, x)
param = svm_parameter('-g 0.25 -c 1')

start = time.time()
m = svm_train(prob, param)
end = time.time()
print("Average Training Time:")
print(end - start)

y, x = svm_read_problem('test.txt')
p_labels, p_acc, p_vals = svm_predict(y, x, m)