'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''

import numpy as np

class NaiveBayes:

    def __init__(self, useLaplaceSmoothing=True):
        '''
        Constructor
        '''
        self.laplacesmoothing=useLaplaceSmoothing
        self.num_classes=None # contains number of classes 
        self.classes=None    # contains all possible classes in data
        self.count_feature_class=None # Contains Count from feature in each class
        self.count_classes= None # contains count of each class in the whole data set
        
    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
        n,d=X.shape
        #print n
        self.classes=np.unique(y)
        self.num_classes=self.classes.size
        self.count_feature_class=np.zeros((self.num_classes,d))      
        self.count_classes=np.zeros(self.num_classes)
        for i in range(self.num_classes):
            get_class=(y==self.classes[i]) # Get which index is class[i]
            self.count_classes[i]=(get_class).sum() # count of class[i]
            for j in range(n):
                if (get_class[j]):# for each class[i]
                   self.count_feature_class[i,:]=self.count_feature_class[i,:]+X[j]# accumulate the count
        #print self.count_feature_class
        for i in range(self.num_classes):
            Temp_feat_count=self.count_feature_class[i,:]
            crr=0
            if(self.laplacesmoothing):
               crr=1
            Temp_feat_count=Temp_feat_count+crr
            #print self.count_feature_class[i,:]
            #print Temp_feat_count
            self.count_feature_class[i,:]=Temp_feat_count/(self.count_feature_class[i,:].sum()+d)  # probability of theta_cj , number of       
        #print self.count_classes
        self.count_classes=self.count_classes/float(self.count_classes.sum()) # this give P(theta_c), probability of class given documents in dataset 
        


    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        n,d=X.shape
        #print self.count_classes
        p_thetac =np.log(self.count_classes)
        p_thetacj=X.dot((np.log(self.count_feature_class)).T)
        y=np.zeros((n,self.num_classes))
        for i in range(n):
            y[i,:]=p_thetac+p_thetacj[i,:]  
        #print y[0:10,:]
        return self.classes[np.argmax(y,axis=1)]

    
    def predictProbs(self, X):
        '''
        Used the model to predict a vector of class probabilities for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-by-K numpy array of the predicted class probabilities (for K classes)
        '''
        n,d=X.shape
        
        p_thetac =np.log(self.count_classes)
        p_thetacj=X.dot((np.log(self.count_feature_class)).T)
        y=np.zeros((n,self.num_classes))
        for i in range(n):
            y[i,:]=p_thetac+p_thetacj[i,:] 
        for i in range(n):
            z=y[i,:].max()
            y[i,:]=y[i,:]-z
            exp_y=np.exp(y[i,:])
            y[i,:]=exp_y/exp_y.sum()
        return y        
        
        
        
class OnlineNaiveBayes:

    def __init__(self, useLaplaceSmoothing=True):
        '''
        Constructor
        '''
        self.online_laplacesmoothing=useLaplaceSmoothing
        self.online_num_classes=0 # contains number of classes 
        self.online_classes=[]    # contains all possible classes in data
        self.online_count_feature_class=[] # Contains Count from feature in each class
        self.online_count_classes=[] # contains count of each class in the whole data set
        self.online_cond_feature=[]
        self.online_prob_class=[]

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
        n,d=X.shape
        #print n,d
        found = 0
        if self.online_num_classes==0:
           self.online_classes=np.append([],y[0])
           self.online_num_classes=self.online_num_classes+1
           self.online_count_feature_class=np.zeros((1,d))
           self.online_count_feature_class[0,:]=self.online_count_feature_class[0,:]+X[0,:]
           self.online_count_classes=np.append(self.online_count_classes,1)
           
        else:
            for i in range(0,self.online_num_classes):
                if self.online_classes[i]==y[0]:
                   self.online_count_feature_class[i,:]=self.online_count_feature_class[i,:]+X[0,:]
                   self.online_count_classes[i]=self.online_count_classes[i]+1
                   found=1
            if(found==0):
                self.online_classes=np.append(self.online_classes,y[0])
                self.online_count_feature_class=np.vstack((self.online_count_feature_class,X[0,:]))
                self.online_count_classes=np.append(self.online_count_classes,1)
                self.online_num_classes=self.online_num_classes+1

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        n,d=X.shape
        self.online_cond_feature=np.copy(self.online_count_feature_class)
        #print self.online_count_classes
        for i in range(self.online_num_classes):
            temp_feat_count=self.online_cond_feature[i,:]
            crr=0
            if(self.online_laplacesmoothing):
               crr=1
            temp_feat_count=temp_feat_count+crr
            self.online_cond_feature[i,:]=temp_feat_count/float(self.online_cond_feature[i,:].sum()+d)
        self.online_prob_class=np.copy(self.online_count_classes)
        self.online_prob_class=self.online_prob_class/float(self.online_prob_class.sum())
        p_thetac =np.log(self.online_prob_class)
        p_thetacj=X.dot((np.log(self.online_cond_feature)).T)
        #print self.online_count_feature_class
        y=np.zeros((n,self.online_num_classes))
        for i in range(n):
            y[i,:]=p_thetac+p_thetacj[i,:] 
        #print np.argmax(y,axis=1)
        return self.online_classes[np.argmax(y,axis=1)]    
    
    def predictProbs(self, X):
        '''
        Used the model to predict a vector of class probabilities for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-by-K numpy array of the predicted class probabilities (for K classes)
        '''
        n,d=X.shape
        
        self.online_cond_feature=np.copy(self.online_count_feature_class)
        #print self.online_count_classes
        for i in range(self.online_num_classes):
            temp_feat_count=self.online_cond_feature[i,:]
            crr=0
            if(self.online_laplacesmoothing):
               crr=1
            temp_feat_count=temp_feat_count+crr
            self.online_cond_feature[i,:]=temp_feat_count/(self.online_cond_feature[i,:].sum()+d)
        self.online_prob_class=np.copy(self.online_count_classes)
        self.online_prob_class=self.online_prob_class/(self.online_prob_class.sum())
        p_thetac =np.log(self.online_prob_class)
        p_thetacj=X.dot((np.log(self.online_cond_feature)).T)
        
        y=np.zeros((n,self.online_num_classes))
        for i in range(n):
            y[i,:]=p_thetac+p_thetacj[i,:] 
        for i in range(n):
            z=y[i,:].max()
            y[i,:]=y[i,:]-z
            exp_y=np.exp(y[i,:])
            y[i,:]=exp_y/exp_y.sum()
        return y  