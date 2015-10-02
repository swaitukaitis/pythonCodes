class NN(object):
    
    def __init__(self, inputLayerSize, hiddenLayerSize, numLabels):
        self.ils=inputLayerSize
        self.hls=hiddenLayerSize
        self.nl=numLabels
  
    def sig(self, z):
        return(1./(1.+np.exp(-z)))
    
    def sigGrad(self, z):
        return(self.sig(z)*(1.-self.sig(z)))
    
    def unrollTheta(self, arr1, arr2):
        return(np.concatenate([arr1.reshape(np.size(arr1), order='F'), arr2.reshape(np.size(arr2), order='F')]))
        
    def rollTheta(self, arr):
        arr1=np.reshape(arr[0:(self.ils+1)*self.hls], [self.hls, self.ils+1], order='F')
        arr2=np.reshape(arr[(self.ils+1)*self.hls:], [self.nl,self.hls+1] , order='F')
        return(arr1, arr2)
    
    def seedParams(self):
        epsilon=0.1
        return np.random.uniform(-epsilon, epsilon, (self.ils+1)*(self.hls)+(self.hls+1)*(self.nl))
    
    def splitSet(self, X, y, f=[0.6, 0.2, 0.2]):
        n=y.size
        ind=np.arange(n)
        np.random.shuffle(ind)
        X, y=X[ind,:], y[ind]
        Xt, yt=X[0:round(n*f[0]),:], y[0:round(n*f[0])]
        Xv, yv=X[round(n*f[0]):round(n*(f[0]+f[1])),:], y[round(n*f[0]):round(n*(f[0]+f[1]))]
        Xe, ye=X[round(n*(f[0]+f[1])):round(n*(f[0]+f[1]+f[2])),:], y[round(n*(f[0]+f[1])):round(n*(f[0]+f[1]+f[2]))]
        return(Xt, yt, Xv, yv, Xe, ye)
    
    def costFunction(self, nn_params, *args):
        X, y, lam=args
        m=np.float(X.shape[0])
        theta1, theta2 = self.rollTheta(nn_params)
        theta1_grad=np.zeros(theta1.shape)
        theta2_grad=np.zeros(theta2.shape)
        J=0.0
        #This loop does the prop and back prop for each training example i
        for i in np.arange(m):
            if self.nl>1:
                yvec=np.zeros(self.nl)
                yvec[y[i]]=1.0
            else:
                yvec=np.float(y[i])
            a1=np.hstack([1.0, X[i,:]])
            z2=np.dot(theta1, a1)
            a2=np.hstack([1.0, self.sig(z2)])
            z3=np.dot(theta2, a2)
            a3=self.sig(z3)
            J+=1./m*(np.dot(-yvec, np.log(a3))-np.dot((1.-yvec), np.log(1.-a3)))
            d3=a3-yvec
            theta2_grad=theta2_grad+1./m*np.outer(d3, a2.T)
            d2=np.dot(theta2[:,1:].T, d3)*net.sigGrad(z2)
            theta1_grad=theta1_grad+1./m*np.outer(d2, a1.T)

        #Now add the regularization terms
        add1=copy(theta1)
        add1[:,0]=0.0
        theta1_grad=theta1_grad+lam/m*add1

        add2=copy(theta2)
        add2[:,0]=0.0
        theta2_grad=theta2_grad+lam/m*add2

        J=J+lam/(2.*m)*( (theta1[:,1:]*theta1[:,1:]).sum()+(theta2[:,1:]*theta2[:,1:]).sum() ) 
        grad=self.unrollTheta(theta1_grad, theta2_grad)
        
        self._funcVal=J
        
        return( [J, grad] )
    
    def callbackFn(self, Xi):
        print '#: '+str(self.Nfeval)+'  J: '+str(self._funcVal), '\r',
        self.Nfeval += 1
    
    def train(self, X, y, lam, maxiter=100, gtol=1e-6, disp=False):
        initialGuess = self.seedParams()
        self.Nfeval=1
        self.valCost=np.array([])
        options = {'maxiter': maxiter, 'disp': disp, 'gtol': gtol}
        _res = optimize.minimize(self.costFunction, initialGuess, jac=True, method='CG', 
                                 args=(X, y, lam), options=options, callback=self.callbackFn)
        
        self.res=_res
        self.theta1, self.theta2 = self.rollTheta(_res.x)
    
    def predict(self, X):
        a1=np.hstack([ones([X.shape[0],1]), X])
        z2=np.dot(self.theta1, a1.T)
        a2=np.vstack([ones([1, z2.shape[1]]), self.sig(z2)])
        z3=np.dot(self.theta2, a2)
        a3=self.sig(z3)
        return np.argmax(a3, axis=0)
    
    def accuracy(self, X , y):
        acc=sum(y==net.predict(X), dtype='float')/X.shape[0]
        print 'Well all right!  The accuracy of the test set predictions is:  '+str(acc)
        return(acc)
        