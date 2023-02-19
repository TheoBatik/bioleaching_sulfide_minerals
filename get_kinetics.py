#------------------------------------------------------------------------------------------
# ESTIMATING THE KINETIC PARAMETERS FOR SULFIDE MINERAL BIOLEACHING SYSTEM
#------------------------------------------------------------------------------------------

# Setup
import numpy as np
from src.rate_optimiser import RateOptimiser
from src.utils import load_csv, fetch_initial_states

ropter = RateOptimiser( reversible=True )
ropter.set_system()
ropter.input( 
    load_csv( csv_name='dataset_1_updated' ),
    fetch_initial_states( ropter.species, csv_name='c0_base' ) 
) 
#------------------------------------------------------------------------------------------

# Checks 
print('species_m', ropter.species_m)
print('states_m', ropter.states_m)
print('species', ropter.species)
print('i hidden states', ropter.ihs)
print('states_nm', ropter.states_nm)


#------------------------------------------------------------------------------------------

# Optimisation

optimal_k = ropter.optimise( n_epochs=1, n_hops=2 )
print( optimal_k )
times = np.linspace( 0, int(ropter.eval_times[-1]), 100 )
# sorted(np.concatenate((np.linspace(0, 503), np.logspace(-14, 1))))


# All quantities
ropter.save_results(
    # eval_times=times,
    predicted=True, 
    measured=True,
    plot_name='(all quantities)'
)

# All measured quantities
ropter.save_results(
    # eval_times=times,
    ignore=[ s for s in ropter.species if s not in ropter.species_m ],
    predicted=True, 
    measured=True,
    plot_name='(all measured quantities)'
)

# Individual plots
for element in ropter.species:
    ropter.save_results(
        # eval_times=times,
        ignore=[ s for s in ropter.species if s not in [ element ] ],
        predicted=True, 
        measured=True,
        plot_name='(' + element + ')'
    )

# Individual plots prediction only
for element in ropter.species:
    ropter.save_results(
        # eval_times=times,
        ignore=[ s for s in ropter.species if s not in [ element ] ],
        predicted=True, 
        measured=False,
        plot_name='( prediction only, ' + element + ')'
)

# Total iron
ropter.plot_total_Fe(
    # eval_times=times,
    # ignore=[ s for s in ropter.species if s not in ropter.species_m ],
    predicted=True,
    measured=True,
    plot_name='(total iron)'
)

# Total iron (prediciton only)
ropter.plot_total_Fe(
    # eval_times=times,
    # ignore=[ s for s in ropter.species if s not in ropter.species_m ],
    predicted=True,
    measured=False,
    plot_name='(predicted iron)'
)


# Print each reaction
for rxn in ropter.reaction_system.rxns:
    print(rxn)