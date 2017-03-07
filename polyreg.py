import numpy as np


#-----------------------------------------------------------------
#  Class PolynomialRegression
#-----------------------------------------------------------------

class PolynomialRegression:
    Theta=[]
    #Theta_NP_Array=[]
    #mean = 0
    #variance=0

    def __init__(self, degree = 1, regLambda = 1E-8):
        '''
        Constructor
        '''
        self.degree = degree
        self.regLambda = regLambda
        Theta=list()
        for i in range(degree):
            Theta.append(0)
        self.Theta_NP_Array=np.array(Theta)


    def polyfeatures(self, X, degree):
        '''
        Expands the given X into an n * d array of polynomial features of
            degree d.

        Returns:
            A n-by-d numpy array, with each row comprising of
            X, X * X, X ** 3, ... up to the dth power of X.
            Note that the returned matrix will not inlude the zero-th power.

        Arguments:
            X is an n-by-1 column numpy array
            degree is a positive integer
        '''

        n=len(X)
        expn=list()
        for i in range(n):
            expx=list()
            for j in range(1,degree+1,1):
                x=X[i]
                expx.append(x**j)
            expn.append(expx)
        expn=np.array(expn)
        return expn
        

    def fit(self, X, y):
        '''
            Trains the model
            Arguments:
                X is a n-by-1 array
                y is an n-by-1 array
            Returns:
                No return value
            Note:
                You need to apply polynomial expansion and scaling
                at first
        '''

        #print X
        Xex=self.polyfeatures(X,self.degree)
        #print Xex
        self.mean=np.mean(Xex,axis=0)
        #print "Mean = ",self.mean
        self.variance=np.std(Xex,axis=0)
        #print "standard varance = " ,self.variance
        std_Xex=(Xex-self.mean)/self.variance
        #print "Standardized = ",std_Xex
        n,d=Xex.shape
        std_Xex=np.c_[np.ones(n),std_Xex]
        n,d=Xex.shape
        #d=d-1
        regMatrix=self.regLambda*np.eye(d+1)
        regMatrix[0,0]=0
        #self.meany=np.mean(y,axis=0)
        #self.variancey=np.std(y,axis=0)
        #y_transformed=(y-self.meany)/self.variancey
        self.Theta_NP_Array=np.linalg.pinv(std_Xex.T.dot(std_Xex) + regMatrix).dot(std_Xex.T).dot(y)
   
        
        
        
    def predict(self, X):
        '''
        Use the trained model to predict values for each instance in X
        Arguments:
            X is a n-by-1 numpy array
        Returns:
            an n-by-1 numpy array of the predictions
        '''

        n=len(X)
        Xex=self.polyfeatures(X,self.degree)
        Xex=(Xex-self.mean)/self.variance
        Xex=np.c_[np.ones(n),Xex]
        return np.dot(Xex,self.Theta_NP_Array.T)



#-----------------------------------------------------------------
#  End of Class PolynomialRegression
#-----------------------------------------------------------------



def learningCurve(Xtrain, Ytrain, Xtest, Ytest, regLambda, degree):
    '''
    Compute learning curve
        
    Arguments:
        Xtrain -- Training X, n-by-1 matrix
        Ytrain -- Training y, n-by-1 matrix
        Xtest -- Testing X, m-by-1 matrix
        Ytest -- Testing Y, m-by-1 matrix
        regLambda -- regularization factor
        degree -- polynomial degree
        
    Returns:
        errorTrain -- errorTrain[i] is the training accuracy using
        model trained by Xtrain[0:(i+1)]
        errorTest -- errorTrain[i] is the testing accuracy using
        model trained by Xtrain[0:(i+1)]
        
    Note:
        errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    '''
    
    n = len(Xtrain);
    m=len(Ytest)
    errorTrain = np.zeros((n))
    errorTest = np.zeros((n))
    polyd=PolynomialRegression(degree,regLambda)
    for i in range(2,n,1):
        polyd.fit(Xtrain[0:i],Ytrain[0:i])
        errorTrain[i-1]=(np.sum((Ytrain[0:i]-polyd.predict(Xtrain[0:i]))**2))/(i)
        errorTest[i-1]=(np.sum((Ytest-polyd.predict(Xtest))**2))/(m)
    polyd.fit(Xtrain[0:n],Ytrain[0:n])
    errorTrain[n-1]=(np.sum((Ytrain[0:n]-polyd.predict(Xtrain[0:n]))**2))/(n)
    errorTest[n-1]=(np.sum((Ytest-polyd.predict(Xtest))**2))/(m)
    
   
    return (errorTrain, errorTest)
