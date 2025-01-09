

class PPinvWrapper:
    """
    Wrapper for P and Pinv pair
    
    methods:
    fit(X, X2d=None) : fit P and Pinv
    transform(X) : transform X to 2D
    inverse_transform(X) : inverse transform 2D to X
    """

    def __init__(self, P, Pinv):
        
        self.P = P
        self.Pinv = Pinv
    def __call__(self, X):
        return self.P(X)
    def transform(self, X):
        return self.P.transform(X)
    # with keywrod argumentscon
    def inverse_transform(self, X, **kwargs):
        return self.Pinv.transform(X, **kwargs)

    def fit(self, X, X2d=None, **kwargs):
        if X2d is None:
            self.X2d = self.P.fit_transform(X)
        else:
            self.X2d = X2d

        ## check is **kwargs is required for Pinv
        if 'y' in kwargs:
            self.Pinv.fit(self.X2d, X, **kwargs)
        else:
            self.Pinv.fit(self.X2d, X)
        return self
    