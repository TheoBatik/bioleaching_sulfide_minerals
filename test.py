from chempy import ReactionSystem  # The rate constants below are arbitrary

class ReactionSystemExtended( ReactionSystem ):
    
    def update_rate_params( self ):
        pass

rsys = ReactionSystemExtended.from_string("""2 Fe+2 + H2O2 -> 2 Fe+3 + 2 OH-; 42
    2 Fe+3 + H2O2 -> 2 Fe+2 + O2 + 2 H+; 17
    H+ + OH- -> H2O; 1e10
    H2O -> H+ + OH-; 1e-4""")  # "[H2O]" = 1.0 (actually 55.4 at RT)

from chempy.kinetics.ode import get_odesys
odesys, extra = get_odesys(rsys)
print(odesys)
from collections import defaultdict
import numpy as np
tout = sorted(np.concatenate((np.linspace(0, 23), np.logspace(-8, 1))))
print(type(tout))
c0 = defaultdict(float, {'Fe+2': 0.05, 'H2O2': 0.1, 'H2O': 1.0, 'H+': 1e-2, 'OH-': 1e-12})
print('\n Initial States' )
print(c0)
print(c0.keys())
result = odesys.integrate(tout, c0, atol=1e-12, rtol=1e-14)
print(result)
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax in axes:
    _ = result.plot(names=[k for k in rsys.substances if k != 'H2O'], ax=ax)
    _ = ax.legend(loc='best', prop={'size': 9})
    _ = ax.set_xlabel('Time')
    _ = ax.set_ylabel('Concentration')
_ = axes[1].set_ylim([1e-13, 1e-1])
_ = axes[1].set_xscale('log')
_ = axes[1].set_yscale('log')
_ = fig.tight_layout()
_ = fig.savefig('kinetics.png', dpi=72)

