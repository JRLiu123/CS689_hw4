
import numpy as np
from bagging_voting import *
from NN_final2 import *

if __name__ == '__main__':
    nor = True
    file_path = 'hw3_house_votes_84.csv'
    structure = [32,3]
    k = 10
    dataset = Bagging_voting(file_path,k,normalization = nor)
    whole_data,D1,D2 = dataset.readfile()
    reg  = 0.05
    num_iter = 10
    alpha = 0.1
    Accuracy_test = []
    F1_test = []   
    for index in range(k):
        print('#############################################')
        print('This is the '+str(index)+'-th loop.')

        #stratified
        test,train = dataset.stratified(D1,D2,index)  

        Neural_network = NN(structure,reg,num_iter,input_dim = test.shape[1]-1,alpha = alpha,weights = None,title = 'voting')
        D_list,error,accuracy,f1,accuracy2,f1_2 = Neural_network.train(train,test)
        Accuracy_test.append(accuracy2)
        F1_test.append(f1_2)
      
    print('*************************************************')
    print('The average accuracy of test dataset is ',"%.4f" % (np.mean(Accuracy_test)))
    print('The average F1 score of test dataset is ',"%.4f" % (np.mean(F1_test)))
'''
if __name__ == '__main__':
    nor = False
    file_path = 'hw3_house_votes_84.csv'
    structure = [8,4,2]
    k = 10
    dataset = Bagging_voting(file_path,k,normalization = nor)
    whole_data,D1,D2= dataset.readfile()
    reg = 0.05
    num_iter = 5000
    alpha = 0.005
    Accuracy_test = []
    F1_test = []
    for index in range(k):
        print('#############################################')
        print('This is the '+str(index)+'-th loop.')

        #stratified
        test,train = dataset.stratified(D1,D2,index)  

      
        Neural_network = NN(structure,reg,num_iter,input_dim = test.shape[1]-1,alpha = alpha,weights = None,title = 'votes')
        D_list,error,accuracy,f1,accuracy2,f1_2 = Neural_network.train(train,test)
        Accuracy_test.append(accuracy2)
        F1_test.append(f1_2)
      
    print('*************************************************')
    print('The average accuracy of test dataset is ',"%.4f" % (np.mean(Accuracy_test)))
    print('The average F1 score of test dataset is ',"%.4f" % (np.mean(F1_test)))
    #Neural_network.error_plot(error,index,ifsave = True)
'''

'''    
if __name__ == '__main__':
    nor = False
    file_path = 'hw3_house_votes_84.csv'
    data = pd.read_csv(file_path)
    num_samples = int(len(data)*0.1)
    data = data.to_numpy(dtype=float)
    N = range(len(data))
    structure = [8,4,3]
    reg  = 0.05
    num_iter = 500
    alpha = 0.1
    J_list = []
    for i in range(39):
        print('*************************************')
        index = (i+1)*10
        m = index + num_samples
        
        B = data[np.random.choice(data.shape[0], size = m, replace=False), :]
        train = B[:index]
        test = B[index:]
        weights = []


        print('train_shape',train.shape)
        Neural_network = NN(structure,reg,num_iter,input_dim = test.shape[1]-1,alpha = alpha,weights=None,title = 'votes')
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


if __name__ == '__main__':
    nor = True
    file_path = 'hw3_wine.csv'
    structure = [32,3]
    k = 10
    dataset = Bagging_wine(file_path,k,normalization = nor)
    whole_data,D1,D2,D3 = dataset.readfile()
    reg  = 0.05
    num_iter = 1000
    alpha = 0.1
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
