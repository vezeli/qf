import numpy as np

import model

# f :: (a, a, a, a, a, a, a) -> a
k    = lambda xs: xs[0]
m    = lambda xs: xs[1]
sig  = lambda xs: xs[2]
r    = lambda xs: xs[3]
t    = lambda xs: xs[4]
tend = lambda xs: xs[5]
dt   = lambda xs: xs[6]

# g :: (a, a, a, a) -> a
s    = lambda xs: xs[0]
c    = lambda xs: xs[1]
dcds = lambda xs: xs[2]
v    = lambda xs: xs[3]

per_year = lambda x: round(x/365, 5)


def increment_time(xs):
    return k(xs), m(xs), sig(xs), r(xs), t(xs)+dt(xs), tend(xs), dt(xs)


def append(xs, ys):
    if np.size(xs) == 0:
        return np.array([ys])
    else:
        return np.vstack([xs, ys])


def gen_stonk_gbm(s, xs):
    args = m(xs), sig(xs), dt(xs)
    yield s
    yield from gen_stonk_gbm(s + model.ds(s, *args), xs)


def evaluate_bsm(s, xs):
    args = s, k(xs), sig(xs), r(xs), t(xs), tend(xs)
    return model.c(*args), model.delta(*args)


def bsm(s, xs):
    c, d = evaluate_bsm(s, xs)
    return s, c, d


def tail(rs):
    try:
        return rs[-1]
    except IndexError:
        return np.array([0, 0, 0, 0])


def cash_balance(rn, rnm, xs):
    sn, cn, dn = rn
    _, _, dnm, vnm = rnm
    vn = (dn - dnm)*sn + vnm*np.e**(r(xs)*dt(xs))
    return sn, cn, dn, vn


def compute(s, xs, rs):
    v = cash_balance(bsm(s, xs), tail(rs), xs)
    return v


def run(gs, xs, rs):
    if t(xs) > tend(xs):
        return rs
    else:
        r = compute(next(gs), xs, rs)
        rs = run(gs, increment_time(xs), append(rs, r))
        return rs


def start(s0, xs):
    gs = gen_stonk_gbm(s0, xs)
    result = run(gs, xs, np.array([]))
    return np.round(result, 2)
