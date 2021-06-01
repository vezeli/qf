P&L of Dynamically Hedged Option Strategy with Black-Scholes-Merton (BSM) Model

Run ``python example.py`` to generate results.

Note that the model is built using functional recursion. For longer
simulations with large number of time steps the Python interpreter
raises ``RecursionError``. Use ``sys.setrecursionlimit`` to increase
recursion depth.

Requirements
============

The code is written for Python 3.5+ and requires the following packages:

* ``numpy``
* ``matplotlib``
* ``scipy``

To install these packages on Linux or MacOS system run ``pip install
numpy matplotlib scipy`` from the terminal.

Modules
=======

* ``simulation.py`` P&L simulation of dynamic delta hedging strategy

* ``model.py`` BSM model for pricing European call options and geometric
  Brownian motion for underlying stock price

Usage
=====

The project can be executed for educational purpose in the given form
``python example.py``. Alternatively, the modules contain public
functions that can be used to create different hedging strategies and
simulations.

The code is open source and licensed under the terms of Apache license
(see LICENSE).
