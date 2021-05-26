"""
Black-Scholes-Merton expression for pricing call option contracts.
"""
import numpy as np
from scipy.stats import norm


def tau(t, T):
    return T - t


def d1(s, k, v, r, t, T):
    tau_ = max(tau(t, T), 0.001)
    return (np.log(s/k)+(r+v**2/2)*tau_)/v/np.sqrt(tau_)


def d2(s, k, v, r, t, T):
    tau_ = tau(t, T)
    return d1(s, k, v, r, t, T) - v*np.sqrt(tau_)


def c(s, k, v, r, t, T):
    tau_ = tau(t, T)
    return s*norm.cdf(d1(s, k, v, r, t, T)) - k*np.e**(-r*tau_)*norm.cdf(d2(s, k, v, r, t, T))


def delta(s, k, v, r, t, T):
    return norm.cdf(d1(s, k, v, r, t, T))


def vega(s, k, v, r, t, T):
    tau_ = tau(t, T)
    return s*np.sqrt(tau_)/np.sqrt(2*np.pi)*np.e**(-d1(s, k, v, r, t, T)**2/2)


def kappa(s, k, v, r, t):
    return vega(s, k, v, r, t, T)/2/v


def ds(s, m, v, dt):
    return m*s*dt + v*s*np.random.normal()
