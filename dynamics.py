import numpy as np

import core

# arguments = (k, m, v, r, t, tend, dt)

per_year = lambda x: round(x/365, 5)

k = lambda xs: xs[0]
m = lambda xs: xs[1]
v = lambda xs: xs[2]
r = lambda xs: xs[3]
t = lambda xs: xs[4]
tend = lambda xs: xs[5]
dt = lambda xs: xs[6]


def increment_time(xs):
    return k(xs), m(xs), v(xs), r(xs), t(xs)+dt(xs), tend(xs), dt(xs)


def append(xs, ys):
    if np.size(xs) == 0:
        return np.array(ys)
    else:
        return np.vstack([xs, ys])


def gen_stonk_price(s, xs):
    args = m(xs), v(xs), dt(xs)
    yield s + core.ds(s, *args)
    yield from gen_stonk_price(s + core.ds(s, *args), xs)


def evaluate_bsm_model(s, xs):
    args = s, k(xs), v(xs), r(xs), t(xs), tend(xs)
    return core.c(*args), core.delta(*args)


def compute(s, xs):
    c, dcds = evaluate_bsm_model(s, xs)
    p = portfolio(s, c, dcds)
    return s, c, p


def portfolio(s, c, dcds):
    return dcds*s - c


def run(gs, xs, rs):
    if t(xs) > tend(xs):
        return rs
    else:
        r = compute(next(gs), xs)
        rs = run(gs, increment_time(xs), append(rs, r))
        return rs


def start(s0, xs):
    gs = gen_stonk_price(s0, xs)
    result = run(gs, xs, np.array([]))
    return np.round(result, 2)
