from dataclasses import dataclass

import numpy as np

import core


@dataclass
class SimParams:
    k: float
    std: float
    mu: float
    r: float
    n: int
    dt: float

    def __post_init__(self):
        self.ndt = self.n*self.dt


def ds(s, p):
    return core.gbm(s, p.mu, p.std, p.dt)


def increment(ss, cs, ps, par):
    if (tn:=len(ss)) > par.n:
        return ss, cs, ps
    else:
        s = ss[-1] + ds(ss[-1], par)
        c = price_c(s, par, tn-1)
        p = portfolio(s, c)
        ss, cs, ps = increment(
            np.append(ss, s) , np.append(cs, c), np.append(ps, p), par
            )
        return ss, cs, ps


def price_c(s, p, tn):
    return core.c(s, p.k, p.std, p.r, tn*p.dt, p.ndt)


def per_year(x):
    return x/365


def portfolio(s, c):
    return s - c


def s0(s):
    return np.array([s])


#p = SimParams(k=10, std_s=0.03, mu_s=0.2, r=0.05, n=365, dt=per_year(1))
