import numpy as np
import matplotlib.pyplot as plt
from Layers import *
class NN(object):
    def __init__(self,structure,reg,num_iter,input_dim,alpha,weights,title):
        self.structure = structure
        self.num_iter = num_iter
        
        self.input_dim = input_dim
        self.weights = weights if weights is not None else []
        for i in range(len(self.structure)):
            w = np.random.random((self.structure[i], self.input_dim+1))*2-1
            self.weights.append(w) 
            
            self.input_dim = structure[i]

        self.alpha = alpha    
        self.reg = reg
        self.title =title
        

    def y_array(self,y):
        y_array = []
        for i in range(self.structure[-1]):
            if y == i:
                y_array.append(1)
            if y != i:
                y_array.append(0)
        y_array = np.array(y_array)
        
        return np.matrix(y_array).reshape(-1,1)
    
    def loss(self,train):
        D_list = []
        J_list = []

        right_num = 0
        y_pre_list = []
        y_true_list = []

        for ins in range(len(train)): 
            #print('This is the '+str(ins)+'-th instance.')
            X = train[ins]#ins]
            
            
            if (self.title == 'wine'):
                x = np.array(X[1:])
                y = np.array(X[0])-1
            else:
                x = np.array(X[:-1])
                y = np.array(X[-1])
            
            layer1 = Layer(self.structure[0],len(x),x,weights=self.weights[0],add_bias= True)
            layer_list = []
            layer_list.append(layer1)        
            a1,pair1 = layer1.forwardPropogate()

            self.pair_list = []
            activ_list = []
            activ_list.append(a1)
            self.pair_list.append(pair1)

            for i in range(len(self.structure)-1):
                layer_name = 'theta'+str(i+2)
                layer_name = Layer(self.structure[i+1],self.structure[i],a1,weights= self.weights[i+1],add_bias= True)
                layer_list.append(layer_name)
                a,pair = layer_name.forwardPropogate()
                #print('a'+str(i+2)+':',a)
                a1 = a
                activ_list.append(a)
                self.pair_list.append(pair)   
            
            delta_list = []
            y_array = self.y_array(y)
            
       
            delta_pre =  activ_list[-1] - y_array
            delta_list.append(delta_pre)


            for i in range(len(self.structure)-1,0,-1):
                # calculate delta
                
                delta_name = 'delta'+str(i)
                delta_name =  layer_list[i].backPropogate(delta_pre,self.pair_list[i])
               
                delta_list.append(delta_name) 
                delta_pre = delta_name
            
            J_normal = self.normal_cost(y_array,activ_list[-1])
            J_list.append(J_normal)
            
            delta_pre =  activ_list[-1] - y_array
            D = {}
            for i in range(len(self.structure)-1,-1,-1):
                #calculate gradient
               
                d = self.gradient(delta_pre,self.pair_list[i])
          
                D[f"theta{i+1}"] = d

                if (i!=0):
                    delta_pre = delta_list[len(self.structure)-i]
            D_list.append(D)
           
            y_pre = np.argmax(activ_list[-1])

            y_pre_list.append(y_pre)
            
            y_true_list.append(y)
           
            if y_pre == y:
                right_num = right_num + 1 
       
      
        accuracy = right_num / len(train)

        pre_list = []
        rec_list = []
        
        for i in range(self.structure[-1]):
            TP = FN = FP = TN = 0
            
            for j in range(len(train)):
            
                if y_true_list[j] == i and y_pre_list[j] == i:
                
                    TP = TP + 1
                    
                if y_true_list[j] == i and y_pre_list[j] != i:
                
                    FN = FN + 1
                    
                    
                if y_true_list[j] != i and y_pre_list[j] == i:
                
                    FP = FP + 1
                    
                if y_true_list[j] != i and y_pre_list[j] != i:
                
                
                    TN = TN + 1

            if TP + FP == 0:
                Pre = 0
            else:
                Pre = 1.0 * TP / (TP + FP)

            if TP + FN == 0:
                Rec = 0
            else:
                Rec = 1.0 * TP / (TP + FN)

            pre_list.append(Pre)

            rec_list.append(Rec)

        avg_pre = np.mean(pre_list)

        avg_rec = np.mean(rec_list)


        f1 = (2 * avg_pre * avg_rec) / (avg_pre + avg_rec)
        
        return D_list,J_list,accuracy,f1
    
    def gradient(self,delta,pair):

        a = pair[0]
        d = np.dot(delta,a.T)
        
        return d      
    
    def regularizer(self,pair):
        
        w = pair[1]
        P = self.reg*w
        P[:,0] = 0

        return P
    def update(self,D_list):
            
        
        for i in range(len(self.structure)):
            
            key = f'theta{i+1}'

            

            gradient = np.zeros_like(D_list[0][key])

            for j in range(len(D_list)):
                gradient = gradient + D_list[j][key]
                
            P = self.regularizer(self.pair_list[i])

            D = (gradient + P) / len(D_list)

            self.weights[i] = self.weights[i] - self.alpha * D


        return self.weights     
    
    def normal_cost(self,y_true,y_pre):
 
        J = []

        for i in range(len(y_true)):

            j = -y_true[i,0]*np.log(y_pre[i,0])-(1-y_true[i,0])*np.log(1-y_pre[i,0])
            J.append(j)
        J = np.sum(J)

        return J
    
    def regular_item(self,train):
        S = 0
        for i in range (len(self.weights)):
            a = np.array(self.weights[i]).T[1:]
            s = np.power(a,2)
            S = S + np.sum(s)

        S = self.reg/2/len(train)*S

        return S
    
    def cost(self,J_list,train):
        J = np.mean(J_list)
        S = self.regular_item(train)
        return J + S
    
 
    def train(self,train,test):
        error = []
        error1 = []

        for i in range(self.num_iter):
            

            D_list,J_list,accuracy,f1 = self.loss(train)
            D1_list1,J_list1,accuracy1,f1_1 = self.loss(test)
            
            J = self.cost(J_list,train)
           
            J1 = self.cost(J_list1,train)

           
            error.append(J)
            error1.append(J1)

            self.weights = self.update(D_list) 
    
     
        return D_list,error1,accuracy,f1,accuracy1,f1_1

        
    def error_plot(self,error,index,ifsave = None,title = 'test'):
        x_ = np.arange(0,len(error))
        plt.plot(x_, error) 
        if (ifsave==True):
            plt.savefig(title+'J_'+str(index+1)+'-th.png')
        plt.show()
        #plt.close()

        
        
    
    
