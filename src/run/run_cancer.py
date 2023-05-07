import numpy as np
from bagging_cancer import *
from NN_final2 import *

if __name__ == '__main__':
    nor = False
    file_path = 'datasets/hw3_cancer.csv'
    structure = [8,2]
    k = 10
    dataset = Bagging_cancer(file_path,k,normalization = nor)
    whole_data,D1,D2= dataset.readfile()
    reg = 0.05
    num_iter = 10
    alpha = 0.1
    Accuracy_test = []
    F1_test = []
    e = 0.001
    for index in range(k):
        print('#############################################')
        print('This is the '+str(index)+'-th loop.')

        #stratified
        test,train = dataset.stratified(D1,D2,index)  

      
        Neural_network = NN(structure,reg,num_iter,input_dim = test.shape[1]-1,alpha = alpha,weights = None,title = 'cancer')
        D_list,error,accuracy,f1,accuracy2,f1_2 = Neural_network.train(train,test)
        Accuracy_test.append(accuracy2)
        F1_test.append(f1_2)
      
    print('*************************************************')
    print('The average accuracy of test dataset is ',"%.4f" % (np.mean(Accuracy_test)))
    print('The average F1 score of test dataset is ',"%.4f" % (np.mean(F1_test)))
    #Neural_network.error_plot(error,index,ifsave = True)
'''
if __name__ == '__main__':
    nor = False
    file_path = 'hw3_cancer.csv'
    structure = [8,2]
    k = 10
    dataset = Bagging_cancer(file_path,k,normalization = nor)
    whole_data,D1,D2= dataset.readfile()
    reg = 0
    num_iter = 500
    alpha = 0.1
    Accuracy_test = []

    F1_test = []
    e = 0.001
    J_list = []
    error_list = []
    for n in range(69):
        print('This is the '+str(n+1)+'trainset.')
        #Accuracy_test = []
        #F1_test = []
        for index in range(k):
            print('#############################################')
            print('This is the '+str(index)+'-th loop.')
            i2 = (n+1)*10
            #stratified
            test,train = dataset.stratified(D1,D2,2)  
            train = train[:i2]
            weights = []
            input_dim = test.shape[1]-1
            for i in range(len(structure)):
                np.random.seed(0)
                w = np.random.random((structure[i], input_dim+1))*2-1
                weights.append(w) 
                input_dim = structure[i]
            

        print('train_shape',train.shape)
        Neural_network = NN(structure,reg,num_iter,input_dim = test.shape[1]-1,alpha = alpha,weights=weights,title = 'cancer')
        D_list,error= Neural_network.train(train,test)#,accuracy,f1,accuracy2,f1_2 = Neural_network.train(train,test)
        #Accuracy_test.append(accuracy2)
        #F1_test.append(f1_2)
        print('error:',error[-1])
        J_list.append(error[-1])
        #Neural_network.error_plot(error,index,ifsave = True)
    print('************************************')
        #print('The average accuracy of test dataset is ',"%.4f" % (np.mean(Accuracy_test)))
        #print('The average F1 score of test dataset is ',"%.4f" % (np.mean(F1_test)))
    print('J:',error)
    print('error',J_list)
'''
