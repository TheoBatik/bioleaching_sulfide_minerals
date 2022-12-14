#------------------------------------------------------------------------------------------
# ESTIMATING THE KINETIC PARAMETERS FOR SULFIDE MINERAL BIOLEACHING SYSTEM
#------------------------------------------------------------------------------------------

# Temp / Tests
import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------------------

# Setup

from src.rate_optimiser import RateOptimiser
from src.utils import load_csv, fetch_initial_states

ropter = RateOptimiser()
ropter.set_system( reversible=False )
ropter.input( 
    load_csv(),
    fetch_initial_states( ropter.species ) 
) 

#------------------------------------------------------------------------------------------

# Optimisation

optimal_k = ropter.optimise( )
print( optimal_k )



# print(a.Results)

# Plot solution
# minerals = ['Pentlandite', f'Pyrrhotite_{x}', 'Pyrite', 'Chalcopyrite']
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# for ax in axes:
#     a.plot(names=[s for s in ropter.reaction_system.substances if s != 'H20'], ax=ax)
#     ax.legend(loc='best', prop={'size': 9})
#     ax.set_xlabel('Time')
#     ax.set_ylabel('Concentration')
# axes[0].title.set_text('Unscaled')
# axes[0].get_legend().remove()
# axes[1].set(ylabel='')
# axes[1].title.set_text('Log scale')
# # axes[1].set_ylim([0.1, 3])
# axes[1].set_xlim([10e-5, ropter.eval_times])
# axes[1].set_xscale('log')
# axes[1].set_yscale('log')
# axes[1].legend( bbox_to_anchor=(0.7, 0.1, 1, 0.5), loc='lower center', borderaxespad=0 ) # plt.legend(bbox_to_anchor=(1.02, 0.1), loc='upper left', borderaxespad=0)
# fig.suptitle('Concentration of all species over time', fontsize=16)
# fig.tight_layout()
# fig.savefig('results/plots/TEST.png', dpi=72)

# print( ko.__dict__ )
# t_out = np.logspace(-8, 1/4) #sorted(np.concatenate((np.linspace(0, t_end), np.logspace(-8, -3))))
# print(t_out)
# Override t_out
# ko.t_out = t_out

# ode_system, _ = get_odesys( reaction_system )

# Solve ODE system
# states_predicted = ode_system.integrate(
#     t_out, 
#     initial_states, 
#     atol=1e-12, 
#     rtol=1e-14
#     )
# print( ko.objective() )


# visible_states = 'Cu Ni Fe'

#------------------------------------------------------------------------------------------

# Optimise
# solution = ode_system.integrate(t_out, c0, atol=1e-12, rtol=1e-14)



# set initial k attribute

# LOOP
#     Create ODE system (reaction rate expression called to fetch k automatically)
#     BASINHOP ( Objective( k , ode_system ) ) => k_new
#     Update k -> k_new












# ode_new = update_reaction_rates( reaction_system, 1)

# with open(file) as file:
# n_epochs = 1
# max_epoch = 5
# while n_epochs < max_epoch:
#     pass
    # ret = basinHop( objective( kinetics ) )
    # write ret.x, ret.fun
    # update_kinetics

    # basinhopping(self.objective, self.fetch_random_k_expo(), minimizer_kwargs=self.minimizer_kwargs, T=T,
    #                     niter=n_iters, disp=display, take_step=self.custom_hop, callback=None)


# get solution
# objective( solution )
# 

# from chempy.kinetics.ode import get_odesys


# ode_system, _ = get_odesys( ropter.reaction_system )
# solution = ode_system.integrate(ropter.eval_times, ropter.states_0, atol=1e-12, rtol=1e-13)

# # Plot solution
# # minerals = ['Pentlandite', f'Pyrrhotite_{x}', 'Pyrite', 'Chalcopyrite']
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# for ax in axes:
#     solution.plot(names=[s for s in ropter.reaction_system.substances if s != 'H20'], ax=ax)
#     ax.legend(loc='best', prop={'size': 9})
#     ax.set_xlabel('Time')
#     ax.set_ylabel('Concentration')
# axes[0].title.set_text('Unscaled')
# axes[0].get_legend().remove()
# axes[1].set(ylabel='')
# axes[1].title.set_text('Log scale')
# # axes[1].set_ylim([0.1, 3])
# axes[1].set_xlim([10e-5, t_end])
# axes[1].set_xscale('log')
# axes[1].set_yscale('log')
# axes[1].legend( bbox_to_anchor=(0.7, 0.1, 1, 0.5), loc='lower center', borderaxespad=0 ) # plt.legend(bbox_to_anchor=(1.02, 0.1), loc='upper left', borderaxespad=0)
# fig.suptitle('Concentration of all species over time', fontsize=16)
# fig.tight_layout()
# fig.savefig('results/plots/TEST.png', dpi=72)