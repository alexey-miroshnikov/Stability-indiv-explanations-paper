############################################## 
# Data generating model and auxiliary methods
############################################## 
# Code authors:
#   Alexey Miroshnikov
#   Konstandinos Kotsiopoulos
# Consultant:
# 	Khashayar Filom
############################################## 
# version 1 (June 2024)
# packages:
#	numpy 1.22.4
#	xgboost 1.7.5
#   scikit-learn 1.2.2
###############################################

from copy import deepcopy
import matplotlib.pyplot as plt; plt.style.use('ggplot')
import numpy as np
from sklearn  import ensemble
import xgboost as xgb

##################################################################### 
# build model method
##################################################################### 

def build_model(pred,resp,model_type,model_params):

    ml_model_constr_dict = { "XGBoost": xgb.XGBRegressor,
                             "GBM": ensemble.GradientBoostingRegressor,
                             "RF" : ensemble.RandomForestRegressor   }

    if model_type in ml_model_constr_dict:
        regr = ml_model_constr_dict[model_type](**model_params)
        regr.fit(pred,resp)
    else:
        raise ValueError("This type of models does not exist")

    return regr

##################################################################### 
# auxiliary methods
##################################################################### 

def L2_norm_rv(X,*,axis=None):

    assert isinstance(X,np.ndarray)
    assert len(X.shape)<=2

    if len(X.shape)==1:
        return np.sqrt(np.mean(np.power(X,2)))

    if len(X.shape)==2:		
        if axis is not None:
            assert axis>=0
            assert axis<=1
            return np.sqrt(np.mean(np.power(X,2),axis=axis))
        else:
            return np.sqrt(np.mean(np.power(X,2),axis=axis))

def L2_norm_vec(X,*,axis=None):

    assert isinstance(X,np.ndarray)
    assert len(X.shape)<=2

    if len(X.shape)==1:
        return np.sqrt(np.sum(np.power(X,2)))

    if len(X.shape)==2:		
        if axis is not None:
            assert axis>=0
            assert axis<=1
            return np.sqrt(np.sum(np.power(X,2),axis=axis))
        else:
            return np.sqrt(np.sum(np.power(X,2),axis=axis))

##################################################################### 
# a class for the true regressor f_true(x0,x1,x2) = 3*x1*x2, which
# includes a method for the marginal Shapley value phi[{0,1,2},v^ME]
##################################################################### 

class true_response_model:
    
    def __init__(self):
        self._pred_dim = 3

    def __call__(self,X):
        return self.predict(X)

    def predict(self,X):
        assert len(X.shape)==2
        assert X.shape[1]==self._pred_dim
        return 3 * X[:,1] * X[:,2]

    # X = samples to explain, X_ave = background dataset
    def shapley_value(self, X, X_ave):
        assert len(X.shape)==2
        assert X.shape[1]==self._pred_dim
        X_mean = np.mean(X_ave,axis=0)
        prod_mean = np.mean(X_ave[:,1] * X_ave[:,2])        
        vals = np.zeros( shape = X.shape )        
        vals[:,1] = (3/2)*(X[:,1]*X[:,2]-X[:,2]*X_mean[1]+X[:,1]*X_mean[2]-prod_mean)
        vals[:,2] = (3/2)*(X[:,1]*X[:,2]-X[:,1]*X_mean[2]+X[:,2]*X_mean[1]-prod_mean)
        return vals

    @property
    def pred_dim(self):
        return self._pred_dim

#############################################################################
## Class for generating data (X,Y) with Y = f_true(x) + eps 
#############################################################################
 
class data_generator_model:

    def __init__(self, **argdict):

        self._argdict = deepcopy(argdict)
        
        self._eps_pred = argdict.get("eps_pred", None )

        self._eps_regr = argdict.get("eps_regr", None )
                
        self.true_response = true_response_model()

        self._pred_dim = 3
        
        assert self._pred_dim == self.true_response.pred_dim

        if self._eps_pred is None:
            self._eps_pred = 0.0

        if self._eps_regr is None:
            self._eps_regr = 0.0


    def sampler_X(self, size, seed=None ):

        if seed is not None:
            np.random.seed( seed = seed )

        X = np.zeros(shape=(size,self._pred_dim))

        gap = self._argdict.get("gap", None)

        if gap is None:
            gap = 1.0
        else:
            assert gap>0.0
            assert gap<2.0
        
        if size>0:
            Z = np.random.uniform(-1,1,size=size)
            X[:,0] = Z + np.random.normal( 0.0, scale = self._eps_pred, size = size )
            X[:,1] = np.sqrt(2) * np.sin( np.pi * Z * 0.25 ) \
                     + np.random.normal(0.0, scale = self._eps_pred, size = size )
            coin=np.random.binomial(1,0.5,size=size)
            Z0=np.random.uniform( -1.0,  -(gap/2), size=size )
            Z1=np.random.uniform(  (gap/2),   1.0, size=size )
            X[:,2] = (1-coin) * Z0 + coin*Z1

        return X

    def true_response_model(self,X):			
        return self.true_response(X)

    def regr_noise(self, size, seed=None):
        if seed is not None:
            np.random.seed( seed = seed )
        return np.random.uniform( -self._eps_regr, self._eps_regr, size=size )

    def dataset_sampler(self, size, seed=None):
        X  = self.sampler_X(size=size, seed=seed )
        Y  = self.true_response_model(X) + self.regr_noise(size)
        return [X,Y]

    @property
    def pred_dim(self):        
        return self.pred_dim