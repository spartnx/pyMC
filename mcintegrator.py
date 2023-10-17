from inspect import signature
import numpy as np
import time
import matplotlib.pyplot as plt

#%% Monte Carlo Integrator
class MonteCarloIntegrator(object):
    """Class wrapping several Monte Carlo techniques to integrate functions of multiple variables.
        Inputs:
            > fcn: function to integrate
            > seed: integer to initialize random number generator
    """
    def __init__(self, fcn, seed=0):
        # Inputs
        self.rng = np.random.Generator(np.random.MT19937(seed))
        self.fcn = fcn
        sig = signature(self.fcn)
        self.dim = sig.parameters['dim'].default
        if type(sig.parameters['lbs'].default) == list:
            self.lbs = np.array(sig.parameters['lbs'].default)
            assert len(self.lbs)==self.dim, "The number of lower bounds 'lbs' must be equal to 'dim'."
        else:
            self.lbs = None
        if type(sig.parameters['ubs'].default) == list:
            self.ubs = np.array(sig.parameters['ubs'].default)
            assert len(self.ubs)==self.dim, "The number of upper bounds 'ubs' must be equal to 'dim'."
        else:
            self.ubs = None
        # Outputs
        self.last_run = None # MC method used in the last run
        self.n = None # Number of samples
        self.fcn_samples = None # data
        self.mean = None # sample mean
        self.var = None # sample variance
        self.se = None # standard error estimate
        self.re = None # relative error estimate
        self.time = None # compute time, sec
        return
    
    def compute_mean(self, data=[]):
        if len(data)==0:
            assert self.last_run!=None, "First run a MC integration method."
            self.mean = np.mean(self.fcn_samples)
            return self.mean
        else:
            return np.mean(data)
    
    def compute_variance(self, data=[]):
        if len(data)==0:
            assert self.last_run!=None, "First run a MC integration method."
            self.var = np.var(self.fcn_samples, ddof=1)
            return self.var
        else:
            return np.var(data, ddof=1)
    
    def compute_se(self): # se = standard error
        assert self.last_run!=None, "First run a MC integration method."
        self.se = np.sqrt(self.var / self.n)
        return
    
    def compute_re(self): # re = relative error
        assert self.last_run!=None, "First run a MC integration method."
        self.re = self.se / abs(self.mean)
        return

    def transformed_fcn(self, U):
        """Integrand after a change of variables bring the bounds of the integral to 0 and 1."""
        X = self.lbs + (self.ubs-self.lbs)*U
        coeff = np.prod(self.ubs-self.lbs)
        return self.fcn(X)*coeff
    
    def fcn_2d(self, x, y, fcn, shape):
        X = np.concatenate((x.reshape((-1,1)), y.reshape((-1,1))), axis=1)
        z = fcn(X).reshape(shape)
        return z
    
    def display(self):
        print(f"\nMethod: {self.last_run}")
        print(f"Number of samples: {self.n}")
        print(f"  > sample mean: {self.mean}")
        print(f"  > sample variance: {self.var}")
        print(f"  > standard error: {self.se}")
        print(f"  > relative error: {self.re}")
        print(f"  > compute time (overall): {self.time} sec")
        print(f"  > compute time (per sample): {self.time/self.n} sec\n")
        return
    
    def surface_plot(self, plot_original=True, square=True):
        """Surface plot of the square of the function to integrate"""
        assert self.dim==2, "This method only works for functions of two variables."
        if plot_original:
            fcn = self.fcn
            lbs = self.lbs
            ubs = self.ubs
        else:
            fcn = self.transformed_fcn
            lbs = np.zeros(len(self.lbs))
            ubs = np.ones(len(self.ubs))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = np.arange(lbs[0], ubs[0], 0.05)
        y = np.arange(lbs[1], ubs[1], 0.05)
        X, Y = np.meshgrid(x, y)
        if square:
            ZF = self.fcn_2d(np.ravel(X), np.ravel(Y), fcn, X.shape)**2
        else:
            ZF = self.fcn_2d(np.ravel(X), np.ravel(Y), fcn, X.shape)
        ax.plot_surface(X, Y, ZF)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
        return
        
    def mc_uniform(self, n=int(1e6), antithetic=False):
        """Integrate the desired function after a change of variables bringing the 
        bounds to 0 and 1. Sampling is from the uniform(0,1) distribution.

        Inputs:
            > n: number of samples to draw from uniform(0,1) for each input to the function
            > antithetic: whether to use the method of antithetic veraibles or not

        Outputs:
            > estimate of the variance of transformed_fcn
            > time per sample
        """
        self.n = n
        t0 = time.time()
        if antithetic==False:
            self.last_run = "MC uniform"
            U = self.rng.uniform(size=(n,self.dim))
            self.fcn_samples = self.transformed_fcn(U)
        else:
            self.last_run = "MC uniform (antithetic variables)"
            U = self.rng.uniform(size=(n,self.dim))
            V = 1 - U
            self.fcn_samples = 0.5*(self.transformed_fcn(U) + self.transformed_fcn(V))
        # Record data
        self.compute_mean()
        self.compute_variance()
        self.compute_se()
        self.compute_re()
        self.time = time.time() - t0 # sec
        return self.var, self.time/self.n

    def mc_general_wo_antithetic(self, w, Finv, n, original_func):
        """MC general without using method of antithetic variables.
        """
        self.last_run = "MC general (without antithetic variables)"
        # Sample uniformly distributed variates
        U = self.rng.uniform(size=(n,self.dim)) 
        # Sample X from desired distributions sepcified in w (inverse transform method)
        if type(Finv[0]) == tuple: # if the distribution is a Scipy distribution
            dist = Finv[0][0]
            params = Finv[0][1:]
            X = dist.rvs(*(params), size=n).reshape((-1,1))
        else: # if the distribution is user defined
            X = Finv[0](U[:,0]).reshape((-1,1))
        for i in range(1,self.dim):
            if type(Finv[i]) == tuple:
                dist = Finv[i][0]
                params = Finv[i][1:]
                new_col = dist.rvs(*(params), size=n).reshape((-1,1))
            else:
                new_col = Finv[i](U[:,i]).reshape((-1,1))
            X = np.concatenate((X, new_col), axis=1)
        # Apply each distribution to each column of X
        if type(w[0]) == tuple:
            dist = w[0][0]
            params = w[0][1:]
            W = dist.pdf(X[:,0], *(params)).reshape((-1,1))
        else:
            W = w[0](X[:,0]).reshape((-1,1))
        for i in range(1,self.dim):
            if type(w[i]) == tuple:
                dist = w[i][0]
                params = w[i][1:]
                new_col = dist.pdf(X[:,i], *(params)).reshape((-1,1))
            else:
                new_col = w[i](X[:,i]).reshape((-1,1))
            W = np.concatenate((W, new_col), axis=1)
        # Compute product of columns of W (to obtain joint densities)
        joint_pdf = np.prod(W, axis=1)
        # Evaluate function to integrate at point X
        if original_func == False:
            F = self.transformed_fcn(X)
        else:
            F = self.fcn(X)
        # Record data
        if (joint_pdf == 1).all():
            self.fcn_samples = F
        else:
            self.fcn_samples = F/joint_pdf
        return 
    
    def mc_general_w_antithetic(self, w, Finv, n, original_func):
        """MC general using method of antithetic variables.
        """
        self.last_run = "MC general (with antithetic variables)"
        # Sample uniformly distributed variates (U) and their antithetic counterparts (V)
        U = self.rng.uniform(size=(n,self.dim)) 
        V = 1 - U
        # Sample X, Y from desired distributions sepcified in w
        if type(Finv[0]) == tuple:
            raise Exception("Inverse cdfs only in input (MC generalantithetic)!")
        else:
            X = Finv[0](U[:,0]).reshape((-1,1))
            Y = Finv[0](V[:,0]).reshape((-1,1))
        for i in range(1,self.dim):
            if type(Finv[i]) == tuple:
                raise Exception("Inverse cdfs only in input (MC generalantithetic)!")
            else:
                new_colX = Finv[i](U[:,i]).reshape((-1,1))
                new_colY = Finv[i](V[:,i]).reshape((-1,1))
            X = np.concatenate((X, new_colX), axis=1)
            Y = np.concatenate((Y, new_colY), axis=1)
        # Apply each distribution to each column of X, Y
        if type(w[0]) == tuple:
            raise Exception("Inverse cdfs only in input (MC generalantithetic)!")
        else:
            Wx = w[0](X[:,0]).reshape((-1,1))
            Wy = w[0](Y[:,0]).reshape((-1,1))
        for i in range(1,self.dim):
            if type(w[i]) == tuple:
                raise Exception("Inverse cdfs only in input (MC generalantithetic)!")
            else:
                new_colX = w[i](X[:,i]).reshape((-1,1))
                new_colY = w[i](Y[:,i]).reshape((-1,1))
            Wx = np.concatenate((Wx, new_colX), axis=1)
            Wy = np.concatenate((Wy, new_colY), axis=1)
        # Compute product of columns of Wx, Wy (to obtain joint densities)
        joint_pdfX = np.prod(Wx, axis=1)
        joint_pdfY = np.prod(Wy, axis=1)
        # Evaluate function to integrate at X, Y
        if original_func == False:
            Fx = self.transformed_fcn(X)
            Fy = self.transformed_fcn(Y)
        else:
            Fx = self.fcn(X)
            Fy = self.fcn(Y)
        # Record data
        if (joint_pdfX == 1).all() and (joint_pdfY == 1).all():
            self.fcn_samples = 0.5*(Fx + Fy) 
        else: 
            self.fcn_samples = 0.5*(Fx/joint_pdfX + Fy/joint_pdfY)
        return 

    def mc_general(self, w, Finv, n=int(1e6), antithetic=False, original_func=False):
        """Integrate the desired function by sampling each function input 
        from the desired distributions. Use method of antithetic variables.
        
        Inputs:
            > w: list of pdfs from which the inputs to the function are sampled
            > Finv: list of inverse cdfs associated with the pdfs in w
            > n: number of samples to draw
            > antithetic: whether to use the method of antithetic variables
            > original_func: whether to use the original function (fcn) or the function obtained after a change of variables (transformed_fcn)

        Outputs:
            > estimate of the variance of the function
            > time per sample
        """
        assert len(w)==self.dim, "Specify a number of densities equal to 'dim'."
        assert len(Finv)==self.dim, "Specify a number of inverse CDFs equal to 'dim'."
        self.n = n
        t0 = time.time()
        if antithetic==False:
            self.mc_general_wo_antithetic(w, Finv, n, original_func)
        else:
            self.mc_general_w_antithetic(w, Finv, n, original_func)
        self.compute_mean()
        self.compute_variance()
        self.compute_se()
        self.compute_re()
        self.time = time.time() - t0
        return self.var, self.time/self.n
    
    def mc_stratified_2D(self, N):
        """Stratified sampling technique for the integration of functions of two variables.
        
        Inputs:
            > N: number of number of samples to draw from each set of the partition of the domain of the integral

        Outputs:
            > estimate of the variance of the function
            > time per sample
        """
        assert len(N.shape)==self.dim, "N must have as many dimensions as 'dim'."
        assert self.dim==2, "This method is only for functions of two variables (for now)."
        self.last_run = "2D stratified MC"
        self.n = sum(N).item(0)
        t0 = time.time()
        r1 = N.shape[0]
        r2 = N.shape[1]
        means = []
        variances = []
        for i in range(1,r1+1):
            for j in range(1,r2+1):
                # Retrieve number of samples to generate in the (i,j) region
                n = N[i-1][j-1]
                # Generate an n-by-2 array of samples uniformly distributed over [0,1]
                U = self.rng.uniform(size=(n,self.dim)) 
                # Transform U's columns into uniformly-distributed samples in the (i,j) region
                lbs = np.array([(i-1)/r1, (j-1)/r2])
                ubs = np.array([i/r1, j/r2])
                X = lbs + (ubs-lbs)*U
                # Evaluate transformed_fcn at X, divided by joint pdf
                F = self.transformed_fcn(X)/(r1*r2)
                # Statistics
                means.append(self.compute_mean(data=F))
                variances.append(self.compute_variance(data=F))
        # outputs
        self.mean = sum(means)
        self.var = sum(variances)
        self.compute_se()
        self.compute_re()
        self.time = time.time() - t0
        return self.var, self.time/self.n
    
    def stratified_sampling_1D(self, N):
        """Stratified sampling technique for the integration of functions of two variables.
        
        Inputs:
            > N: number of number of samples to draw from each set of the partition of the domain of the integral

        Outputs:
            > estimate of the variance of the function
            > time per sample
        """
        assert len(N.shape)==self.dim, "N must have as many dimensions as 'dim'."
        assert self.dim==1, "This method is only for functions of one variable (for now)."
        self.last_run = "1D stratified MC"
        self.n = sum(N).item(0)
        t0 = time.time()
        r1 = N.shape[0]
        means = []
        variances = []
        for i in range(1,r1+1):
            # Retrieve number of samples to generate in the (i,j) region
            n = N[i-1]
            # Generate an n-by-1 array of samples uniformly distributed over [0,1]
            U = self.rng.uniform(size=(n,self.dim)) 
            # Transform U's columns into uniformly-distributed samples in the (i,j) region
            lbs = np.array([(i-1)/r1])
            ubs = np.array([i/r1])
            X = lbs + (ubs-lbs)*U
            # Evaluate transformed_fcn at X, divided by pdf
            F = self.transformed_fcn(X)/(r1)
            # Statistics
            means.append(self.compute_mean(data=F))
            variances.append(self.compute_variance(data=F))
        # outputs
        self.mean = sum(means)
        self.var = sum(variances)
        self.compute_se()
        self.compute_re()
        self.time = time.time() - t0
        return self.var, self.time/self.n


