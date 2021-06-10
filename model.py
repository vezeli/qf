"""
Black-Scholes-Merton model for pricing European call options.
"""
import numpy as np
from scipy.stats import norm

ERR = 0.001


def _tau(t, tend):
    return tend - t


def d1(s, k, v, r, t, tend):
    tau = max(_tau(t, tend), ERR)
    return (np.log(s/k)+(r+v**2/2)*tau)/v/np.sqrt(tau)


def d2(s, k, v, r, t, tend):
    tau = _tau(t, tend)
    return d1(s, k, v, r, t, tend) - v*np.sqrt(tau)


def c(s, k, v, r, t, tend):
    tau = _tau(t, tend)
    n1, n2 = norm.cdf(d1(s, k, v, r, t, tend)), norm.cdf(d2(s, k, v, r, t, tend))
    return s*n1 - k*np.e**(-r*tau)*n2


def delta(s, k, v, r, t, tend):
    return norm.cdf(d1(s, k, v, r, t, tend))


def vega(s, k, v, r, t, tend):
    tau = _tau(t, tend)
    return s*np.sqrt(tau)/np.sqrt(2*np.pi)*np.e**(-d1(s, k, v, r, t, tend)**2/2)


def kappa(s, k, v, r, t, tend):
    return vega(s, k, v, r, t, tend)/2/v


def ds(s, m, v, dt):
    return m*s*dt + v*s*np.random.normal()
