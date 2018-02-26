
import pandas as pd
import pickle
    

class Madelon_data():
    
    def __init__(self):
        ''' define Madelon data files '''   
        #data location 
        self.train='./data/madelon_data/madelon_train.data.txt'
        self.test='./data/madelon_data/madelon_test.data.txt'
        self.valid='./data/madelon_data/madelon_valid.data.txt'
        self.train_labels='./data/madelon_data/madelon_train.labels.txt'
        self.valid_labels='./data/madelon_data/madelon_valid.labels.txt'
        self.params='./data/madelon_data/madelon.param.txt'
       
        
    def load_train(self):
        
        X=pd.read_csv(self.train, delimiter=' ', header=None)
        
        #drop empty feature 500
        X.drop(500,axis=1,inplace=True)
        
        y=pd.read_csv(self.train_labels, delimiter=' ',header=None, names=['target'])
        
        return X , y
    
   
    def load_valid(self):
        
        X=pd.read_csv(self.valid, delimiter=' ', header=None)
        
        #drop empty feature 500
        X.drop(500,axis=1,inplace=True)
        
        y=pd.read_csv(self.valid_labels, delimiter=' ',header=None, names=['target'])
        
        return X , y
    
    def load_parameters(self):
        dfp=pd.read_csv(self.params,skiprows=4, delimiter='\t', names=['Dataset','Pos_ex','Neg_ex','Tot_ex','Check_sum'])
        dfp.set_index('Dataset', inplace=True)
        return dfp
    
    
class Postgres_data():
        
    def __init__(self):
        #RAW DATA PULLED
        self.sample1='./data/pickle_data/Postgres_95_sample1'
        self.sample2='./data/pickle_data/Postgres_95_sample2'
        self.sample3='./data/pickle_data/Postgres_95_sample3'
    

    def load_sample1(self ):
        return pd.read_pickle(self.sample1)
        
    def load_sample2(self ):
        return pd.read_pickle(self.sample2)
        
    def load_sample3(self ):
        return pd.read_pickle(self.sample3)

class Pickle_files():
    
    def __init__(self):
        #DE-SKEWED DATA OUTPUT FROM EDA
        self.datafiles={
            'UCI'     :['./data/pickle_data/UCI_X'          , './data/pickle_data/UCI_y'          ,\
                        './data/pickle_data/UCI_X_train'    , './data/pickle_data/UCI_X_test'    ,
                        './data/pickle_data/UCI_y_train'     , './data/pickle_data/UCI_y_test'     ],
            'Sample1' :['./data/pickle_data/Sample1_X'      , './data/pickle_data/Sample1_y'      ,\
                        './data/pickle_data/Sample1_X_train', './data/pickle_data/Sample1_X_test',\
                        './data/pickle_data/Sample1_y_train' , './data/pickle_data/Sample1_y_test' ],
            'Sample2' :['./data/pickle_data/Sample2_X'      , './data/pickle_data/Sample2_y'      ,\
                        './data/pickle_data/Sample2_X_train', './data/pickle_data/Sample2_X_test',\
                        './data/pickle_data/Sample2_y_train' , './data/pickle_data/Sample2_y_test' ],
            'Sample3' :['./data/pickle_data/Sample3_X'      , './data/pickle_data/Sample3_y'      ,\
                        './data/pickle_data/Sample3_X_train', './data/pickle_data/Sample3_X_test',\
                        './data/pickle_data/Sample3_y_train' , './data/pickle_data/Sample3_y_test' ],
                        }
        self.Samples_bestfeatures_skb='./data/pickle_data/Samples_bestfeatures_skb'
        self.Samples_bestfeatures_sfm='./data/pickle_data/Samples_bestfeatures_sfm'
        self.Samples_bestfeatures_rfe='./data/pickle_data/Samples_bestfeatures_rfe'
        self.Samples_benchmarks='./data/pickle_data/Samples_benchmarks' 
        self.UCI_bestfeatures_skb='./data/pickle_data/UCI_bestfeatures_skb'
        self.UCI_bestfeatures_sfm='./data/pickle_data/UCI_bestfeatures_sfm'
        self.UCI_bestfeatures_rfe='./data/pickle_data/UCI_bestfeatures_rfe'
        self.UCI_benchmarks='./data/pickle_data/UCI_benchmarks' 
        
        
        
    def update_datafiles(self,file):
        self.datafiles=file
        
    def read_datafiles(self):
        #PROVIDE ACCESS TO PICKLE FILE NAMES WHEN REQUIRED- CAN BE USED IN ANY NOTEBOOK
        return self.datafiles
    
    def load_SBF(self):
        #CONSOLDIATE RESULTS FROM FEATURE SELECTION MODELS
        skb=pd.read_pickle(self.Samples_bestfeatures_skb)
        sfm=pd.read_pickle(self.Samples_bestfeatures_sfm)
        #rfe=pd.read_pickle(self.Samples_bestfeatures_rfe)
        df=pd.concat([skb,sfm],axis=0,ignore_index=True)
        return df
    
    def load_UCIBF(self):
        #CONSOLDIATE RESULTS FROM FEATURE SELECTION MODELS
        skb=pd.read_pickle(self.UCI_bestfeatures_skb)
        sfm=pd.read_pickle(self.UCI_bestfeatures_sfm)
        rfe=pd.read_pickle(self.UCI_bestfeatures_rfe)
        df=pd.concat([skb,sfm,rfe],axis=0,ignore_index=True)
        return df
        
    def load_benchmarks(self):
        #PROVIDE CONSOLIDATED VIEW OF BENCHMARKS - NOT USED
        UCI=pd.read_pickle(self.UCI_benchmarks)
        Samples=pd.read_pickle(self.Samples_benchmarks)
        df=pd.concat([UCI,Samples],axis=0,ignore_index=True)
        return df
        
        