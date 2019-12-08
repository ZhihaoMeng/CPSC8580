import os
os.environ["R_HOME"] = "/home/chongm/.conda/envs/ML/lib/R"
os.environ["PATH"] = "$R_HOME/bin:$PATH"
#our customized Gussian mixture model with fused lasso
import numpy as np
from scipy.stats import multivariate_normal

from scipy import io
from rpy2 import robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
r = robjects.r
rpy2.robjects.numpy2ri.activate()

#np.set_printoptions(threshold = 1e6)
importr('genlasso')
importr('gsubfn')

from sklearn.mixture import GaussianMixture
from sklearn.mixture.gaussian_mixture import _estimate_gaussian_parameters, _compute_precision_cholesky

class GMM(GaussianMixture):
    def __init__(self,y, n_components=1, covariance_type='full', tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
                 weights_init=None, means_init=None, precisions_init=None,
                 random_state=None, warm_start=False,
                 verbose=0, verbose_interval=10):
        super(GMM,self).__init__(
            n_components=n_components, tol=tol, reg_covar=reg_covar,
            max_iter=max_iter, n_init=n_init, init_params=init_params,
            random_state=random_state, warm_start=warm_start,
            verbose=verbose, verbose_interval=verbose_interval)
        
        self.Y= y
        
        
    
    """Customized m-step to fit fused lasso"""
    def _m_step(self, X, log_resp):   
        """M step.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        log_resp : array-like, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        n_samples, n_features = X.shape
        self.weights_, self.mu, self.covariances_ = (
            _estimate_gaussian_parameters(X, np.exp(log_resp), self.reg_covar,
                                          self.covariance_type))
        
        # update lasso coefficient
        print "*************updata means by fused lasso now*****************"
        r_ic = np.exp(log_resp)
        
        for i in range(self.n_components):
            idx = np.where(np.argmax(r_ic,axis=1) == i)
            
            print "len(idx):", len(idx[0])
            #ensure it can be fitted by fused lasso
            if len(idx[0])>(n_samples/(2*self.n_components)):
                print "fused lasso used"
                data_X_i = r.matrix(X[idx[0]], nrow = len(idx[0]), ncol = n_features)
                data_Y_i = r.matrix(self.Y[idx[0]],nrow = len(idx[0]), ncol = 1)
                n = r.nrow(data_X_i)
                p = r.ncol(data_X_i)
                print "lasso_n:",n
                print "lasso_p:",p
                results = r.fusedlasso1d(y=data_Y_i, X=data_X_i)
                result = np.array(r.coef(results, np.sqrt(n*np.log(p)))[0])[:,-1]
                mu_i = np.multiply(result,np.mean(data_X_i,axis=0))
                if i == 0:
                    self.means_ = mu_i
                else:
                    self.means_ = np.vstack((self.means_, mu_i))
                
            else:
                print "not enough data for fused lasso"
                if i == 0:
                    self.means_ = self.mu[i]
                else:
                    self.means_ = np.vstack((self.means_,self.mu[i]))
                
        self.weights_ /= n_samples
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type) 
  
