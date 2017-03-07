
import numpy as np
class LogisticRegression:

    def __init__(self, alpha =0.000023, regLambda=0.01, epsilon=0.0001, maxNumIters = 10000):
        '''
        Constructor
        '''
        self.alpha=alpha
        self.regLambda=regLambda
        self.epsilon= epsilon
        self.maxNumIter=maxNumIters
        self.Theta=None

    

    def computeCost(self, theta, X, y, regLambda):
        '''
        Computes the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            a scalar value of the cost  ** make certain you're not returning a 1 x 1 matrix! **
        '''
        n,d=X.shape
        print "XShape  ",X.shape,"ThetaShape ",theta.shape
        theta=(theta).reshape(1,d)
        hthetax=np.dot(X,theta.T)
        logcost=np.log(self.sigmoid(hthetax))
        neglogcost=np.log(1-self.sigmoid(hthetax))
        ylog=np.dot(y.T,logcost)
        negylog=np.dot((1-y).T,neglogcost)
        return -ylog.sum()-negylog.sum()+(((np.square(theta).sum())*regLambda)/2)            

    
    
    def computeGradient(self, theta, X, y, regLambda):
        '''
        Computes the gradient of the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            the gradient, an d-dimensional vector
        '''
        n,d=X.shape
        #print "XShape  ",X.shape,"ThetaShape ",theta.shape
        gradlen=len(theta)
        #print gradlen
        theta=(theta).reshape(1,d)
        hthetax=np.dot(X,theta.T)
        regMatrix=self.regLambda*np.eye(d)
        regMatrix[0,0]=0        
        cost=np.dot((self.sigmoid(hthetax)-y).T,X)
        #print "shape of cost",cost.shape
        return cost +theta.dot(regMatrix)

    


    def sigmoid(self, Z):
        '''
        Computers the sigmoid function 1/(1+exp(-z))
        '''
        return 1/(1+np.exp(-Z))


    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
        '''
        n,d=X.shape
        print "InputXShape  ",X.shape
        self.Theta=np.zeros((1,d+1))
        self.Theta=(self.Theta).reshape(1,d+1)
        X1=np.c_[np.ones((n,1)),X]
        for i in range(self.maxNumIter):
            delta=self.computeGradient(self.Theta,X1,y,self.regLambda)
            #print delta.shape
            if((np.square(delta)).sum()>self.epsilon):
               print i,self.computeCost(self.Theta,X1,y,self.regLambda)
               self.Theta=self.Theta-(self.alpha)*delta
            else:
               break

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
        Returns:
            an n-dimensional numpy vector of the predictions
        '''
        n,d=X.shape
        Z=self.sigmoid(np.dot(np.c_[np.ones((n,1)),X],(self.Theta).T))
        Z[Z>=0.5]=1
        Z[Z<0.5]=0
        #print Z.shape
        return Z
    