import numpy as np
import pandas as pd
from bagging_wine import *
from NN_final2 import *
if __name__ == '__main__':
    nor = True
    file_path = 'hw3_wine.csv'
    structure = [8,3]
    k = 10
    dataset = Bagging_wine(file_path,k,normalization = nor)
    whole_data,D1,D2,D3 = dataset.readfile()
    reg  = 0.01
    num_iter = 10
    alpha = 0.6
    Accuracy_test = []
    F1_test = []   
    for index in range(k):
        print('#############################################')
        print('This is the '+str(index)+'-th loop.')

        #stratified
        test,train = dataset.stratified(D1,D2,D3,index)  

        Neural_network = NN(structure,reg,num_iter,input_dim = test.shape[1]-1,alpha = alpha,weights = None,title = 'wine')
        D_list,error,accuracy,f1,accuracy2,f1_2 = Neural_network.train(train,test)
        Accuracy_test.append(accuracy2)
        F1_test.append(f1_2)
      
    print('*************************************************')
    print('The average accuracy of test dataset is ',"%.4f" % (np.mean(Accuracy_test)))
    print('The average F1 score of test dataset is ',"%.4f" % (np.mean(F1_test)))
    #Neural_network.error_plot(error,index,ifsave = True)

    #x_ = np.arange(0,30)
    #plt.plot(x_*5+5, error_list/30) 
    
    #plt.savefig('J.png')
    #plt.show()
'''
if __name__ == '__main__':
    nor = True
    file_path = 'hw3_wine.csv'
    data = pd.read_csv(file_path,delimiter='\t')
    num_samples = int(len(data)*0.1)
    data = data.to_numpy(dtype=float)
    N = range(len(data))
    structure = [8,3]
    reg  = 0
    num_iter = 500
    alpha = 0.1
    J_list = []
    for i in range(30):
        print('*************************************')
        index = (i+1)*5
        m = index + num_samples
        
        B = data[np.random.choice(data.shape[0], size = m, replace=False), :]
        train = B[:index]
        test = B[index:]
        weights = []
        input_dim = test.shape[1]-1
        for i in range(len(structure)):
            np.random.seed(0)
            w = np.random.random((structure[i], input_dim+1))*2-1
            weights.append(w) 
            input_dim = structure[i]

        print('train_shape',train.shape)
        Neural_network = NN(structure,reg,num_iter,input_dim = test.shape[1]-1,alpha = alpha,weights=weights,title = 'wine')
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


            




    


