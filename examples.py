import numpy as np
from scipy.optimize import newton
from scipy.stats import truncnorm, cosine, norm
import matplotlib.pyplot as plt

from mcintegrator import MonteCarloIntegrator
from samplers import cmset, cm, pmf_estimator, cdf_estimator

#%%  Functions to integrate
# Each function to be integrated by MonteCarloIntegrator must have inputs: X, dim, lbs, ubs, params (optional)
def fcn(X, dim=2, lbs=[0, np.pi/2], ubs=[np.pi/4, 3*np.pi/4]):
    assert X.shape[1]==dim, "The width of the input array X must be equal to 'dim'."
    x = X[:,0]
    y = X[:,1]
    return (x**2)*(y**2)*np.sin(x+y)*np.log(x+y)

def a_func(X, dim=1, lbs=[np.pi/2], ubs=[3*np.pi/4]):
    z = X
    polynom = (1/5)*(z-np.pi/2)**5 - (1/2)*z*(z-np.pi/2)**4 + (1/3)*z**2*(z-np.pi/2)**3
    return polynom*np.sin(z)*np.log(z)

def b_func(X, dim=1, lbs=[3*np.pi/4], ubs=[np.pi]):
    z = X
    polynom1 = (1/5)*(np.pi/4)**5 - (1/2)*z*(np.pi/4)**4 + (1/3)*z**2*(np.pi/4)**3
    polynom2 = (1/5)*(z-3*np.pi/4)**5 -(1/2)*z*(z-3*np.pi/4)**4 + (1/3)*z**2*(z-3*np.pi/4)**3
    return (polynom1 - polynom2)*np.sin(z)*np.log(z)

def european_option(X, dim=1, lbs=None, ubs=None, params=()):
    # Discounted payoff of the European option
    assert X.shape[1]==dim, "The width of the input array X must be equal to 'dim'."
    if len(params) == 0:
        K, S0, T, xi1, xi2 = 100, 90, 0.5, 0.1, 0.2
    else:
        K, S0, T, xi1, xi2 = params
    M = S0*np.exp((xi1-0.5*xi2**2)*T + xi2*np.sqrt(T)*X) - K
    M[M <= 0] = 0
    return np.exp(-xi1*T)*M

#%% Candidate distributions in variable x
def wx1(x): # pdf
    # x in [0,1]
    return 1 - np.cos(np.pi*x)

def Fx1(x): # cdf
    # CDF of w1(x)
    # x in [0,1]
    return x - np.sin(np.pi*x)/np.pi

def Fx1_inv(u): # inverse cdf
    # CDF of w1(x) - u
    # u in [0,1]
    def f(x):
        return Fx1(x) - u
    # Newton algorithm to solve F(x) = u for x
    x = newton(f, 0.5)
    return x

def wx2(x, beta): 
    return (beta/(np.exp(beta)-1)) * np.exp(beta*x)

def Fx2_inv(u, beta):
    return np.log(u*(np.exp(beta)-1) + 1)/beta

def wx_europ_optn(x): 
    return np.ones(len(x))

def Fx_inv_europ_opt(u): 
    return norm.ppf(u)

#%% Candidate distributions in variable y
def wy1(y): # pdf
    return 2*(1-y)

def Fy1(y): # cdf
    return y*(2-y)

def Fy1_inv(u): # inverse cdf
    return 1 - np.sqrt(1 - u)

def wy2(y):
    return np.ones(len(y))

def Fy2(y):
    return y

def Fy2_inv(u):
    return u

def wy3(y, beta):
    return (beta/(np.exp(beta)-1)) * np.exp(beta*(1-y))

def Fy3_inv(u, beta):
    return -np.log(-u*(1 - np.exp(-beta)) + 1)/beta

#%% Distribution to sample from
def pmf(k, l=5, N=20):
    assert (k>=0) and (k<=20), f"k must be in [0..{N}]"
    assert type(k)==int, "k must be integer"
    numerator = l**k / np.math.factorial(k)
    denominator = sum([l**i/np.math.factorial(i) for i in range(N+1)])
    return numerator / denominator

def cdf(k, l=5, N=20):
    assert (k>=0) and (k<=20), f"k must be in [0..{N}]"
    # assert type(k)==int, "k must be integer"
    return sum([pmf(i, l, N) for i in range(k+1)])


#%%
if __name__ == "__main__":
    # Part to run
    part = 9

    if part == 1: # Standard Monte Carlo
        # Define a MC integrator object (1 integrator per function)
        mc = MonteCarloIntegrator(fcn)

        # Standard Monte Carlo (using uniform distribution over function domain)
        var, t = mc.mc_uniform(n=2**20)
        mc.display()

        # Visualize the square of the function to integrate
        mc.surface_plot(plot_original=False) 

    elif part == 2: # Importance sampling
        # Define a MC integrator object (1 integrator per function)
        mc = MonteCarloIntegrator(fcn)

        # Visualize the square of the function to integrate
        mc.surface_plot(plot_original=False) 
        
        # Standard Monte Carlo (using uniform distribution over function domain)
        var_crude, t_crude = mc.mc_uniform(n=2**20)
        mc.display()

        # wA = [wx1, wy1]
        # FinvA = [np.vectorize(Fx1_inv), Fy1_inv]
        # varA, tA = mc.mc_general(wA, FinvA, n=2**20)
        # CR_A, VR_A, ER_A = t_crude/tA, var_crude/varA, t_crude*var_crude/(tA*varA)
        # print(CR_A, VR_A, ER_A)

        # wB = [wx1, wy2]
        # FinvB = [np.vectorize(Fx1_inv), Fy2_inv]
        # varB, tB = mc.mc_general(wB, FinvB, n=2**20)
        # CR_B, VR_B, ER_B = t_crude/tB, var_crude/varB, t_crude*var_crude/(tB*varB)
        # print(CR_B, VR_B, ER_B)

        loc = 0.6 # for truncated normal
        scale = 0.5 # for truncated normal
        a = -loc/scale # for truncated normal
        b = (1 - loc)/scale # for truncated normal
        wC = [(truncnorm, a, b, loc, scale), wy1]
        FinvC = [(truncnorm, a, b, loc, scale), Fy1_inv]
        varC, tC = mc.mc_general(wC, FinvC, n=2**20)
        CR_C, VR_C, ER_C = t_crude/tC, var_crude/varC, t_crude*var_crude/(tC*varC)
        print(CR_C, VR_C, ER_C)

        loc = 0.6 # for truncated normal
        scale = 0.5 # for truncated normal
        a = -loc/scale # for truncated normal
        b = (1 - loc)/scale # for truncated normal
        wD = [(truncnorm, a, b, loc, scale), wy2]
        FinvD = [(truncnorm, a, b, loc, scale), Fy2_inv]
        varD, tD = mc.mc_general(wD, FinvD, n=2**20)
        CR_D, VR_D, ER_D = t_crude/tD, var_crude/varD, t_crude*var_crude/(tD*varD)
        print(CR_D, VR_D, ER_D)

        wE = [(cosine, 0.5, 1/(2*np.pi)), wy1]
        FinvE = [(cosine, 0.5, 1/(2*np.pi)), Fy1_inv]
        varE, tE = mc.mc_general(wE, FinvE, n=2**20)
        CR_E, VR_E, ER_E = t_crude/tE, var_crude/varE, t_crude*var_crude/(tE*varE)
        print(CR_E, VR_E, ER_E)

        wF = [(cosine, 0.5, 1/(2*np.pi)), wy2]
        FinvF = [(cosine, 0.5, 1/(2*np.pi)), Fy2_inv]
        varF, tF = mc.mc_general(wF, FinvF, n=2**20)
        CR_F, VR_F, ER_F = t_crude/tF, var_crude/varF, t_crude*var_crude/(tF*varF)
        print(CR_F, VR_F, ER_F)

        wG = [(truncnorm, a, b, loc, scale), (truncnorm, a, b, loc, scale)]
        FinvG = [(truncnorm, a, b, loc, scale), (truncnorm, a, b, loc, scale)]
        varG, tG = mc.mc_general(wG, FinvG, n=2**20)
        CR_G, VR_G, ER_G = t_crude/tG, var_crude/varG, t_crude*var_crude/(tG*varG)
        print(CR_G, VR_G, ER_G)

        wH = [lambda x: wx2(x,2), lambda y: wy3(y,2)]
        FinvH = [lambda x: Fx2_inv(x,2), lambda y: Fy3_inv(y,2)]
        varH, tH = mc.mc_general(wH, FinvH, n=2**20)
        CR_H, VR_H, ER_H = t_crude/tH, var_crude/varH, t_crude*var_crude/(tH*varH)
        print(CR_H, VR_H, ER_H)

        # The "H" option fares the best - perform a grid search on the parameters
        print("\nParameter search")
        CR_list = []
        VR_list = []
        ER_list = []
        params = []
        for i in np.arange(0.1, 5, 0.1):
            for j in np.arange(0.1, 5, 0.1):
                print(f" > i = {round(i,1)}, j = {round(j,1)}")
                w = [lambda x: wx2(x,i), lambda y: wy3(y,j)]
                Finv = [lambda x: Fx2_inv(x,i), lambda y: Fy3_inv(y,j)]
                var, t = mc.mc_general(w, Finv, n=2**20)
                CR, VR, ER = t_crude/t, var_crude/var, t_crude*var_crude/(t*var)
                CR_list.append(CR)
                VR_list.append(VR)
                ER_list.append(ER)
                params.append((i,j))
                print(f"     {CR}, {VR}, {ER}\n")

        # Retrieve the best parameters for the "H" option and re-evaluate integral
        index_max = np.argmax(ER_list)
        beta1, beta2 = params[index_max]
        ER_max = ER_list[index_max]
        CR_max = CR_list[index_max]
        VR_max = VR_list[index_max] 
        w = [lambda x: wx2(x,beta1), lambda y: wy3(y,beta2)]
        Finv = [lambda x: Fx2_inv(x,beta1), lambda y: Fy3_inv(y,beta2)]
        var, t = mc.mc_general(w, Finv, n=2**20)
        mc.display()
        CR, VR, ER = t_crude/t, var_crude/var, t_crude*var_crude/(t*var)
        print(f"beta1 = {beta1}, beta2 = {beta2}")
        print(f"{CR_max}, {VR_max}, {ER_max}")
        print(f"{CR}, {VR}, {ER}")  

    elif part == 3: # Stratified sampling for Monte Carlo integration
        # Define a MC integrator object (1 integrator per function)
        mc = MonteCarloIntegrator(fcn)

        # Crude Monte Carlo (using uniform distribution over function domain)
        var_crude, t_crude = mc.mc_uniform(n=2**20)
        mc.display()

        # Stratified sampling - r=2**4 subregions
        r = 2**4 
        n = 2**20
        N = np.ones((r,1), dtype=np.int32)*int((n/r))
        var, t = mc.mc_stratified_2D(N)
        mc.display()
        CR, VR, ER = t_crude/t, var_crude/var, t_crude*var_crude/(t*var)
        print(f"{CR}, {VR}, {ER}")  

        # Stratified sampling - r=2**8 subregions
        r = 2**8
        n = 2**20
        N = np.ones((r,1), dtype=np.int32)*int((n/r))
        var, t = mc.mc_stratified_2D(N)
        mc.display()
        CR, VR, ER = t_crude/t, var_crude/var, t_crude*var_crude/(t*var)
        print(f"{CR}, {VR}, {ER}")  

    elif part == 4: # Rao-Blackwellization
        # Standard MC method with uniform distribution
        mc = MonteCarloIntegrator(fcn, seed=0)
        var_crude, t_crude = mc.mc_uniform(n=2**20)
        mc.display()

        # Rao-Blackwell method
        # this method consists in transforming integral(fcn) into lower dimensional integrals
        # integral(fcn) = integral(a_func) + integral(b_func)
        # integral(fcn) is a 2-dimensional integral
        # integral(a_func) and integral(b_func) are 1-dimensional integrals
        mcA = MonteCarloIntegrator(a_func, seed=10)
        mcB = MonteCarloIntegrator(b_func, seed=100)

        r = 2**8
        n = 2**19
        N = np.ones((r,), dtype=np.int32)*int((n/r))

        varA, tA = mcA.stratified_sampling_1D(N)
        mcA.display()

        varB, tB = mcB.stratified_sampling_1D(N)
        mcB.display()

        # Statistics
        mean = mcA.mean + mcB.mean # point estimate of the integral
        var = varA + varB # pooled variance
        se = np.sqrt(var/(2*n))
        re = se/abs(mean)
        t = (tA*n+tB*n)/(2*n)
        print(f"\nCombined A + B")
        print(f"Number of samples: {2*n}")
        print(f"  > sample mean: {mean}")
        print(f"  > sample variance: {var}")
        print(f"  > standard error: {se}")
        print(f"  > relative error: {re}")
        print(f"  > compute time (overall): {t*(2*n)} sec")
        print(f"  > compute time (per sample): {t} sec\n")

        # Performance ratios
        CR, VR, ER = t_crude/t, var_crude/var, t_crude*var_crude/(t*var)
        print(f"{CR}, {VR}, {ER}") 

    elif part == 5: # MC integration with method of antithetic variables
        mc = MonteCarloIntegrator(fcn, seed=0)
        mc.surface_plot(plot_original=False, square=False) 

        # Standard MC sampling without antithetic variables
        var_crude, t_crude = mc.mc_uniform(n=2**20)
        mc.display()

        # Standard MC sampling with antithetic variables
        var1, t1 = mc.mc_uniform(n=2**19, antithetic=True)
        mc.display()
        CR1, VR1, ER1 = t_crude/t1, var_crude/var1, t_crude*var_crude/(t1*var1)
        print(f"{CR1}, {VR1}, {ER1}") 

        # Importance sampling with antithetic variables
        beta1 = 2.6
        beta2 = 0.3
        w = [lambda x: wx2(x,beta1), lambda y: wy3(y,beta2)]
        Finv = [lambda x: Fx2_inv(x,beta1), lambda y: Fy3_inv(y,beta2)]
        var2, t2 = mc.mc_general(w, Finv, n=2**19, antithetic=True)
        mc.display()
        CR2, VR2, ER2 = t_crude/t2, var_crude/var2, t_crude*var_crude/(t2*var2)
        print(f"{CR2}, {VR2}, {ER2}") 

    elif part == 6: # Computation of the expected discounted payoff of the European option
        K_list = [100]
        S0_list = [90, 100, 110]
        T_list = [0.5, 1]
        xi1_list = [0.1]
        xi2_list = [0.2, 0.3, 0.4, 0.5]

        outputs_dict = {}
        for K in K_list:
            outputs_dict[K] = {}
            for S0 in S0_list:
                outputs_dict[K][S0] = {}
                for T in T_list:
                    outputs_dict[K][S0][T] = {}
                    for xi1 in xi1_list:
                        outputs_dict[K][S0][T][xi1] = {}
                        for xi2 in xi2_list:
                            print("\n#######################################")
                            print(f"K = {K}, S0 = {S0}, T = {T}, xi1 = {xi1}, xi2 = {xi2}")
                            params = (K, S0, T, xi1, xi2)
                            def fcn(X, dim=1, lbs=None, ubs=None):
                                return european_option(X, dim=1, lbs=None, ubs=None, params=params)
                            mc = MonteCarloIntegrator(fcn, seed=0)

                            # MC sampling from standard normal distribution without antithetic variable
                            w = [wx_europ_optn]
                            Finv = [Fx_inv_europ_opt]
                            var_crude, t_crude = mc.mc_general(w, Finv, n=int(2e6), original_func=True)
                            mc.display()

                            # MC sampling from standard normal distribution with antithetic variable
                            w = [wx_europ_optn]
                            Finv = [Fx_inv_europ_opt]
                            var, t = mc.mc_general(w, Finv, n=int(1e6), original_func=True, antithetic=True)
                            mc.display()

                            # Performance ratios
                            CR, VR, ER = t_crude/t, var_crude/var, t_crude*var_crude/(t*var)
                            print(f"{CR}, {VR}, {ER}") 

                            # Record data
                            outputs_dict[K][S0][T][xi1][xi2] = {"mean": round(mc.mean, 4),
                                                                "se": round(mc.se, 4),
                                                                "re": round(mc.re, 4),
                                                                "CR": round(CR, 4),
                                                                "VR": round(VR, 4),
                                                                "ER": round(ER, 4)}

        # Output display on the terminal
        K = 100
        xi1 = 0.1
        print("\n####################################################")
        print("VR as a function of xi2 for each (S0, T) combination")  
        print("####################################################")
        for S0 in S0_list:
            for T in T_list:
                VR_list = []
                mean_list = []
                se_list = []
                re_list = []
                for xi2 in xi2_list:
                    VR_list.append(outputs_dict[K][S0][T][xi1][xi2]["VR"])
                    mean_list.append(outputs_dict[K][S0][T][xi1][xi2]["mean"])
                    se_list.append(outputs_dict[K][S0][T][xi1][xi2]["se"])
                    re_list.append(outputs_dict[K][S0][T][xi1][xi2]["re"])
                print(f"(S0, T) = {S0, T}")
                print(f"   > xi2  = {xi2_list}")
                print(f"   > VR   = {VR_list}")
                print(f"   > mean = {mean_list}")
                print(f"   > se   = {se_list}")
                print(f"   > re   = {re_list}\n")

        print("\n####################################################")
        print("VR as a function of T for each (S0, xi2) combination")  
        print("####################################################")
        for S0 in S0_list:
            for xi2 in xi2_list:
                VR_list = []
                mean_list = []
                se_list = []
                re_list = []
                for T in T_list:
                    VR_list.append(outputs_dict[K][S0][T][xi1][xi2]["VR"])
                    mean_list.append(outputs_dict[K][S0][T][xi1][xi2]["mean"])
                    se_list.append(outputs_dict[K][S0][T][xi1][xi2]["se"])
                    re_list.append(outputs_dict[K][S0][T][xi1][xi2]["re"])
                print(f"(S0, xi2) = {S0, xi2}")
                print(f"   > T    = {T_list}")
                print(f"   > VR   = {VR_list}")
                print(f"   > mean = {mean_list}")
                print(f"   > se   = {se_list}")
                print(f"   > re   = {re_list}\n")

        print("\n####################################################")
        print("VR as a function of S0 for each (T, xi2) combination")  
        print("####################################################")
        for T in T_list:
            for xi2 in xi2_list:
                VR_list = []
                mean_list = []
                se_list = []
                re_list = []
                for S0 in S0_list:
                    VR_list.append(outputs_dict[K][S0][T][xi1][xi2]["VR"])
                    mean_list.append(outputs_dict[K][S0][T][xi1][xi2]["mean"])
                    se_list.append(outputs_dict[K][S0][T][xi1][xi2]["se"])
                    re_list.append(outputs_dict[K][S0][T][xi1][xi2]["re"])
                print(f"(T, xi2) = {T, xi2}")
                print(f"   > S0   = {S0_list}")
                print(f"   > VR   = {VR_list}")
                print(f"   > mean = {mean_list}")
                print(f"   > se   = {se_list}")
                print(f"   > re   = {re_list}\n")

    elif part == 7: # Finding the set of cutpoints
        N = 20
        m = 20
        I = cmset(cdf, m, params=(5,N))
        print(f"{I}\n")

    elif part == 8: # Use cutpoint method to sample from a discrete distribution
        N = 20
        m = 20
        Y = cm(cdf, m, n=1e5)
        cdf_true = np.array([cdf(k) for k in range(N+1)])
        pmf_true = np.array([pmf(k) for k in range(N+1)])
        cdf_estimates = np.array([cdf_estimator(k, Y) for k in range(N+1)])
        pmf_estimates = np.array([pmf_estimator(k, Y) for k in range(N+1)])
        mean_estimate = sum([k*pmf_estimates[k] for k in range(N+1)])
        print(abs(cdf_true-cdf_estimates))
        print(abs(pmf_true-pmf_estimates))
        print(mean_estimate)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(cdf_estimates, cdf_true)
        ax.set_xlabel('CDF estimates')
        ax.set_ylabel('True CDf values')
        plt.show()

    elif part == 9: # Use Acceptance-Rejection method to sample from a discrete distribution
        N = 20
        m = 20
        # Visualize pmf
        pmf_true = np.array([pmf(k) for k in range(N+1)])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(range(N+1), pmf_true, marker=".")
        ax.set_xlabel('k')
        ax.set_ylabel('pmf')
        plt.show()
        
        # A-R algorithm
        n = int(1e5)
        pmf_max = max([pmf(k) for k in range(N+1)])
        g = lambda k: pmf(k)/pmf_max
        rng = np.random.Generator(np.random.MT19937(0))
        i = 1
        X = []
        while i <= n:
            U = rng.uniform()
            Y = int(rng.integers(low=0, high=N+1))
            if U <= g(Y):
                X.append(Y)
                i += 1
        X = np.array(X)
        print(max(X), min(X), np.mean(X))

        cdf_true = np.array([cdf(k) for k in range(N+1)])
        pmf_true = np.array([pmf(k) for k in range(N+1)])
        cdf_estimates = np.array([cdf_estimator(k, X) for k in range(N+1)])
        pmf_estimates = np.array([pmf_estimator(k, X) for k in range(N+1)])
        mean_estimate = sum([k*pmf_estimates[k] for k in range(N+1)])
        print(abs(cdf_true-cdf_estimates))
        print(abs(pmf_true-pmf_estimates))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(cdf_estimates, cdf_true)
        ax.set_xlabel('CDF estimates')
        ax.set_ylabel('True CDf values')
        plt.show()



        



        
        

        




        
