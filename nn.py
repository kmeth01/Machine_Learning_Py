import numpy as np
from PIL import Image
import math

class NeuralNet:

    def __init__(self, layers, epsilon=0.12, learningRate=0.1, numEpochs=100):
        '''
        Constructor
        Arguments:
           layers - a numpy array of L-2 integers (L is # layers in the network)
           epsilon - one half the interval around zero for setting the initial weights
           learningRate - the learning rate for backpropagation
           numEpochs - the number of epochs to run during training
        '''
        self.layers=layers
        self.epsilon=epsilon
        self.alpha=learningRate
        self.numEpochs=numEpochs
        #self.labels=0
    def neuron_function(self,z):
        return 1/(1+np.exp(-1*z))
        
    def Forward_Prop(self,A,n):
        for i in range(1,self.n_layers,1):
            temp=np.c_[np.ones((n,1)),A[i-1]]
            z=temp.dot(self.theta[i-1].T)
            A[i]=self.neuron_function(z)
    
    def Backward_Prop_delta(self,A,delta):
        for i in range(self.n_layers-2,0,-1):
            l,h=A[i].shape
            temp=A[i]#np.c_[np.ones((l,1)),A[i]]
            #print delta[i+1].shape , self.theta[i].shape
            delta[i]=np.multiply((np.dot(delta[i+1],self.theta[i]))[:,1:],np.multiply(temp,1-temp))
            
        
    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
        np.random.seed(13)
        n,d=X.shape
        y_convert=[None]*n
        L=len(np.unique(y))
        self.n_layers=len(self.layers)+2
        self.labels=np.unique(y)
        #print self.labels
        # for n layers, there are n-1 theta matrices. 
        #Below list contains the theta matrices. indexed from 0 to n-2 , instead of 1 to n-1
        num_theta_mat=self.n_layers-1
        self.theta=[None]*(num_theta_mat)
        
        # if L is the size of self.layers(doesnt have input and output layer)
        # Hence n = L+2, and there would n-1 theta matrices, therefore number of theta matrices is L+1
        # theta_j is of dimension (s_(j+1),(s_j)+1) : s_j is number of neurons in layer j
        # Below loop intializes from theta_1 to theta_(j-2) j =0 and j-1 
        for i in range(1,num_theta_mat-1,1):
            self.theta[i]=(np.random.uniform(low=-1,high=1,size=(self.layers[i],self.layers[i-1]+1)))*self.epsilon
        #Initializing Theta_0
        self.theta[0]=(np.random.uniform(low=-1,high=1,size=(self.layers[0],d+1)))*self.epsilon
        #print self.theta[0]
        
        if L>2:
           #initializing Theta_L-1
           self.theta[num_theta_mat-1]=(np.random.uniform(low=-1,high=1,size=(L,self.layers[-1]+1)))*self.epsilon
           for i in range(n):
               label=y[i]
               y_convert[i]=np.zeros(L)
               thu=np.where(self.labels==label)
               #print thu[0]
               x=np.asscalar(thu[0])
               y_convert[i][x]=1
               #print y[i],y_convert[i]
        else:
           self.theta[num_theta_mat-1]=(np.random.uniform(low=-1,high=1,size=(1,self.layers[-1]+1)))*self.epsilon
        
        for num in range(self.numEpochs):   
            A=[None]*(self.n_layers)
            A[0]=X
            self.Forward_Prop(A,n)
            delta=[None]*(self.n_layers)
            delta[self.n_layers-1]=A[self.n_layers-1]-y_convert
            self.Backward_Prop_delta(A,delta)
            Grad_theta=[None]*(num_theta_mat)
            for i in range(num_theta_mat):
                l,h=A[i].shape
                temp=np.c_[np.ones((l,1)),A[i]]
                Grad_theta[i]=(1/float(n))*(np.dot((delta[i+1].T)[:,:],temp))
                #print Grad_theta[i].shape
            for i in range(num_theta_mat):
                p,q=self.theta[i].shape
                regmatrix=(np.eye(q))*0.0007
                regmatrix[0][0]=0
                Grad_theta[i]=Grad_theta[i]+np.dot(self.theta[i],regmatrix)
            for i in range(num_theta_mat):
                self.theta[i]=self.theta[i]-self.alpha*Grad_theta[i]
            if(num<2):
              Unrolled=[]
              print "This is  ", num
              for th in range(num_theta_mat):
                  Unrolled.append(self.theta[i].ravel())
              print Unrolled
                  
                
            #print "Theta1",self.theta[1]
            #for l in range(1,self.n_layers-1,1):
            #   print Grad_theta[l].shape

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        n,d=X.shape
        A=[None]*(self.n_layers)
        A[0]=X
        self.Forward_Prop(A,n)
        Z=A[-1]
        #Z[Z>0.5]=1
        for j in range(1):
            for m in range(10):
                m=m+1
                #print (A[-1][m])
        for i in range(n):
            i=i+1
            #print np.argmax(Z[i],axis=0)
        return self.labels[np.argmax(Z,axis=1)]
        
        
    def visualizeHiddenNodes(self, filename):
        '''
        CIS 519 ONLY - outputs a visualization of the hidden layers
        Arguments:
            filename - the filename to store the image
        '''
        Theta_NoBias=[None]*(self.n_layers-1)
        Theta_Images=[None]*(self.n_layers-1)
        for i in range((self.n_layers-1)):
            Theta_NoBias[i]=self.theta[i][:,1:]
        for i in range(len(Theta_NoBias)-1):
            n,d = Theta_NoBias[i].shape
            Theta_Images[i]=[None]*n
            for j in range(n):
                Theta_NoBias[i][j,:]=(Theta_NoBias[i][j,:]-Theta_NoBias[i].min()/(Theta_NoBias[i][j,:].max()-Theta_NoBias[i].min()))*255
                Theta_Images[i][j]=Theta_NoBias[i][j,:].reshape(int(math.sqrt(d)),int(math.sqrt(d)))
                Theta_Images[i][j]=Image.fromarray((Theta_Images[i][j]))
        for i in range(len(Theta_Images)-1):
            q=len(Theta_Images[i])
            w,h=Theta_Images[i][0].size
            blank_image = Image.new("L", (int(math.sqrt(q)*w), int(math.sqrt(q)*h)))
            for j in range(len(Theta_Images[i])):
                #print ((j/int(math.sqrt(q)))*w,(j%int(math.sqrt(q)))*h)
                blank_image.paste(Theta_Images[i][j],((j/int(math.sqrt(q)))*w,(j%int(math.sqrt(q)))*h))
            blank_image.save(str(i)+filename)        
