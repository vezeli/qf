import matplotlib.pyplot as plt

import simulation

PRICE = 10.0 # [$]
STRIKE_PRICE = 10.0 # [$]
DRIFT_RATE = 10.5 # [%]
VOLATILITY = 3.0 # [%]
RISKFREE_RATE = 5.0 # [%/year]
CONTRACT_DURATION = 365 # [days]
TIMESTEP = 1 # [days]

_parameters = simulation.set_parameters(STRIKE_PRICE,
                                        DRIFT_RATE,
                                        VOLATILITY,
                                        RISKFREE_RATE,
                                        CONTRACT_DURATION,
                                        TIMESTEP)

SIMULATIONS = 3
results = (simulation.start(PRICE, _parameters) for _ in range(SIMULATIONS))

fig, (ax1, ax2, ax3) = plt.subplots(3)
for r in results:
    stock_price, call_price, delta, cash_balance = r[:,0], r[:,1], r[:,2], r[:,3]
    ax1.plot(range(len(r)), stock_price)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Stock price")
    ax2.plot(range(len(r)), call_price)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Call price")
    ax3.plot(range(len(r)), call_price - delta*stock_price + cash_balance)
    ax3.set_xlabel("Time")
    ax3.set_ylabel("P&L")
plt.show()
