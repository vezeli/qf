from collections import namedtuple
import numpy as np

Parameters = namedtuple("Parameters", "sigma_s, mu_s, dt")


def per_year(x):
    return x/365


def s0(s):
    return np.array([s])


def params(sigma_s, mu_s, dt):
    return Parameters(sigma_s=sigma_s, mu_s=mu_s, dt=dt) 


def ds(s, p):
    return p.mu_s*s*p.dt + p.sigma_s*s*np.random.normal()


def append(ss, p):
    s = ss[-1]
    v = s + ds(s, p)
    return np.append(ss, v)


def increment(ss, p, n):
    if len(ss) > n:
        return ss
    else:
        ss = increment(append(ss, p), p, n)
        return ss


#s, p = s0(10), params(sigma_s=0.03, mu_s=0.2, dt=per_year(1))
