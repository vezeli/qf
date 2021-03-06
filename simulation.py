from typing import Callable, Tuple

import numpy as np

import model

# types
SIM_PARAMETERS = Tuple[float, float, float, float, float, float, float]
RESULT = Tuple[float, float, float, float]
HF1 = Callable[[SIM_PARAMETERS], float]
HF2 = Callable[[RESULT], float]


def set_parameters(
    k: float, m: float, v: float, r: float, tend: int, dt: int
) -> SIM_PARAMETERS:
    """Formats simulation parameters."""
    rs = map(_percentage, [m, v, r])
    ts = map(_per_year, [tend, dt])
    return (k, *rs, 0, *ts)


def start(initial_price: float, ps: SIM_PARAMETERS) -> np.ndarray:
    """Starts simulation."""
    result = _run(_gen_stonk_gbm(initial_price, ps), ps, np.array([]))
    return np.round(result, 2)


def _gen_stonk_gbm(s, ps):
    args = _m(ps), _sig(ps), _dt(ps)
    yield s
    yield from _gen_stonk_gbm(s + model.ds(s, *args), ps)


def _run(gs, ps, rs):
    if _t(ps) > _tend(ps):
        return rs
    else:
        r = _compute(next(gs), ps, rs)
        rs = _run(gs, _increment_time(ps), _append(rs, r))
        return rs


def _compute(s, ps, rs):
    v = _cash_balance(evaluate_bsm(s, ps), _tail(rs), ps)
    return v


def _cash_balance(rn, rnm, ps):
    sn, cn, dn = rn
    _, _, dnm, vnm = rnm
    vn = (dn - dnm)*sn + vnm*np.e**(_r(ps)*_dt(ps))
    return sn, cn, dn, vn


def evaluate_bsm(s, ps):
    """Evaluates Black-Scholes model for a European call option."""
    args = s, _k(ps), _sig(ps), _r(ps), _t(ps), _tend(ps)
    return s, model.c(*args), model.delta(*args)


_k:    HF1 = lambda ps: ps[0]
_m:    HF1 = lambda ps: ps[1]
_sig:  HF1 = lambda ps: ps[2]
_r:    HF1 = lambda ps: ps[3]
_t:    HF1 = lambda ps: ps[4]
_tend: HF1 = lambda ps: ps[5]
_dt:   HF1 = lambda ps: ps[6]

_s:    HF2 = lambda ps: ps[0]
_c:    HF2 = lambda ps: ps[1]
_dcds: HF2 = lambda ps: ps[2]
_v:    HF2 = lambda ps: ps[3]

_per_year: Callable[[int], float] = lambda x: round(x/365, 5)
_percentage: Callable[[float], float] = lambda x: x/100


def _increment_time(ps):
    return _k(ps), _m(ps), _sig(ps), _r(ps), _t(ps)+_dt(ps), _tend(ps), _dt(ps)


def _append(xs, ys):
    if np.size(xs) == 0:
        return np.array([ys])
    else:
        return np.vstack([xs, ys])


def _tail(rs):
    try:
        return rs[-1]
    except IndexError:
        return np.array([0, 0, 0, 0])
