from dataclasses import dataclass

import numpy as np

import core


@dataclass
class SimParams:
    k: float
    std_s: float
    mu_s: float
    r: float
    n: int
    dt: float

    def __post_init__(self):
        self.ndt = self.n*self.dt


per_year = lambda x: x/365


def s0(s):
    return np.array([s])


def ds(s, p):
    return core.gbm(s, p)


def append(ss, p):
    s = ss[-1]
    v = s + ds(s, p)
    return np.append(ss, v)


def increment(ss, cs, p):
    if len(ss) > p.n:
        return ss, cs
    else:
        c = ss2c(ss, p)
        ss, cs = increment(append(ss, p), np.append(cs, c), p)
        return ss, cs


def ss2c(ss, p):
    s = ss[-1]
    t = -(1-len(ss))*p.dt
    return core.c(s, p.k, p.std_s, p.r, t, p.ndt)


#p = SimParams(k=10, std_s=0.03, mu_s=0.2, r=0.05, n=365, dt=per_year(1))
