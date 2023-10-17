# pyMC
Codebase with implementations of state-of-the-art Monte Carlo sampling and integration techniques. 

The `MonteCarloIntegrator` class in `mcintegrator.py` module implements the following methods:
- Standard Monte Carlo integration with sampling from uniform(0,1)
- General Monte Carlo integration with sampling from user-defined distributions (as long as the inverse cdfs exist)
- Stratified sampling
- Method of antithetic variables

The functions defined in `samplers.py` implement the following methods:
- Cutpoint method for sampling from a discrete distribution

In addition to illustrating the methods outlined above, the `examples.py` module illustrates the following methods:
- Rao-Blackwell method
- Acceptance-Rejection method for sampling from a discrete distribution (the method is valid for sampling from continuous distributions as well)


