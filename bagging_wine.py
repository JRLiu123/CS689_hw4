import numpy as np
import pandas as pd
from sklearn.utils import shuffle


class Bagging_wine:
    
    def __init__(self,file_path,k,normalization = True):

        self.file_path = file_path 
        self.k = k
        self.normalization = normalization
        
    def readfile(self):

        df = pd.read_csv(self.file_path,delimiter='\t')
        
        # unique classes
        classes = np.unique(df.loc[:,'# class'])
        
        # spilt dataset according to classes
        D1 = df.loc[df['# class'] == classes[0]]
        D2 = df.loc[df['# class'] == classes[1]]
        D3 = df.loc[df['# class'] == classes[2]]

        # shuffle the subset
        D1 = shuffle(D1,random_state=0)
        D2 = shuffle(D2,random_state=0)
        D3 = shuffle(D3,random_state=0)
        
        #drop the original index and reset index
        D1 = D1.reset_index(drop=True)
        D2 = D2.reset_index(drop=True)
        D3 = D3.reset_index(drop=True)
        
        return df,D1,D2,D3
 
    def stratified(self,D1,D2,D3,i):
        # the index to split test from train
        index1= np.rint(np.linspace(0, (len(D1)-1), num=self.k+1))
        index2= np.rint(np.linspace(0, (len(D2)-1), num=self.k+1))
        index3= np.rint(np.linspace(0, (len(D3)-1), num=self.k+1))

        i11 = int(index1[i])
        i12 = int(index1[i+1])

        i21 = int(index2[i])
        i22 = int(index2[i+1])
        
        i31 = int(index3[i])
        i32 = int(index3[i+1])
        

        D1_test = D1.loc[i11:i12]
        D1_train1 = D1.loc[0:i11-1]
        D1_train2 = D1.loc[i12+1:]
        

        D2_test = D2.loc[i21:i22]
        D2_train1 = D2.loc[0:i21-1]
        D2_train2 = D2.loc[i22+1:]
 
            
        D3_test = D3.loc[i31:i32]
        D3_train1 = D3.loc[0:i31-1]
        D3_train2 = D3.loc[i32+1:]


        frame_train = [D1_train1, D1_train2, D2_train1, D2_train2, D3_train1, D3_train2]
        train = pd.concat(frame_train)
        train = train.reset_index(drop=True)
        train = shuffle(train)
        
        frame_test = [D1_test,D2_test, D3_test]
        test = pd.concat(frame_test)   
        test = test.reset_index(drop=True)
        test = shuffle(test)
        if self.normalization:
            col_name = test.columns.values.tolist()
            col_name = col_name[1:]

            train_y = train['# class']
            train_X = train.loc[:,col_name]
            train_X = (train_X - train_X.min())/(train_X.max() - train_X.min())
            train = pd.concat([train_y, train_X], axis=1)

            test_y = test['# class']
            test_X = test.loc[:,col_name]
            test_X = (test_X - test_X.min())/(test_X.max() - test_X.min())
            test = pd.concat([test_y, test_X], axis=1)


        test = test.to_numpy(dtype=float)
        train = train.to_numpy(dtype=float) 
        return test,train
   
 
