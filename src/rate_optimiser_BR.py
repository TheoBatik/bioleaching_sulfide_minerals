#------------------------------------------------------------------------------------------

# Imports
from chempy.chemistry import Reaction
from chempy import ReactionSystem
from chempy.kinetics.ode import get_odesys
import numpy as np
from scipy.optimize import basinhopping
import matplotlib.pyplot as plt


#------------------------------------------------------------------------------------------

# Extend the :class:`ReactionSystem` with a method update its own reaction rate params

def update_rate_param( reaction, rate_param ):
    'Updates the reaction rate `param` for single instance of :class:`Reaction`'
    setattr( reaction, 'param', rate_param )
    return reaction

class ReactionSystemExtended( ReactionSystem ):

    def update_rate_params( self, rate_params ):
        '''
        Updates the reaction rate parameters of all :class:`Reaction`s
        in the :class:`ReactionSystem`
        '''

        # Update the rate params of each reaction
        reactions = list( map( update_rate_param, iter(self.rxns), iter(rate_params) ) )
    
        # Update the reactions of the reaction system
        setattr(self, 'rxns', reactions)

#------------------------------------------------------------------------------------------

class CustomBasinHop:
    def __init__(self, stepsize=1):
        self.stepsize = stepsize
    def __call__(self, k):
        s = self.stepsize
        random_step = np.random.uniform(low=-s, high=s, size=k.shape)
        k += random_step
        return k

custom_hop = CustomBasinHop()

#------------------------------------------------------------------------------------------
class RateOptimiser:
    '''
    Methods:
        define the reaction system (:class:`ReactionSystem`)
        load the input attributes (initial conditions & measured states)
        calculate the net error between the measurements & model prediciton (objective function)
        optimise the rate parameters by minimisation of the objective function
    '''
#------------------------------------------------------------------------------------------

    def __init__(self, reversible=True):
        self.reversible = reversible

#------------------------------------------------------------------------------------------
    def set_system( self, reactants, products ):
        '''
        Sets the reaction system (:class:`ReactionSystem`)
        for the hard-coded reactions (:class:`Reaction`)
        '''

        # Number of reactions (one-directional)
        num_rxns = len( reactants )
        setattr( self, 'num_rxns', num_rxns )

        # Forward rate params & reactions
        forward_rate_params = np.random.uniform( low=0.001, high=2, size=num_rxns ) # forward reaction rate params
        # forward_rate_params[-1] = 0.30913341785292603
        forward_reactions = [ Reaction( r, p, k ) for r, p, k in zip( reactants, products, forward_rate_params ) ]
        reactions = forward_reactions

        # Backward rate params & reactions
        if self.reversible:
            backward_rate_params = np.random.uniform( low=0.001, high=0.9, size=num_rxns ) # backward reaction rate params
            backward_reactions = [ Reaction( r, p, k ) for r, p, k in zip( products, reactants, backward_rate_params ) ]
            reactions += backward_reactions
            setattr( self, 'num_rxns', 2*num_rxns )

        # Set reaction system
        species = set().union( *[ rxn.keys() for rxn in reactions ] )
        reaction_system = ReactionSystemExtended( reactions, species )
        substances = reaction_system.substances.keys()
        setattr( self, 'reaction_system', reaction_system )

        # Derive species from reaction system (to correspond to solution of the ODE system)
        species = [ sub for sub in substances ]
        setattr( self, 'species', species)

#------------------------------------------------------------------------------------------

    def input( self, measurements, initial_states ):

        # Set input attributes
        setattr( self, 'states_m', measurements[0][:, 1:] ) # measured states
        setattr( self, 'species_m', measurements[1][1:] ) # measured species
        setattr( self, 'states_0', initial_states ) # initial conditions
        
        # Set times at which to evalutate the solution of the ODE system
        setattr( self, 'eval_times', measurements[0][:, 0] )

        # List the indices of hidden states within those predicted states
        indices_of_hidden_states = [ ]
        for i, s in enumerate( self.species ):
            if s not in self.species_m:
                indices_of_hidden_states.append( i )
        np.asarray( indices_of_hidden_states, dtype=int )
        setattr( self, 'ihs', indices_of_hidden_states )

        # Set maxium of the measured states
        setattr( self, 'max_measured', np.max( self.states_m ) )

        # Normlise the measured states
        setattr( self, 'states_nm', self.states_m / self.max_measured )
        


#------------------------------------------------------------------------------------------


    def objective( self, rate_params_ex ):
        '''
        Returns the `net_error` as the sum (over time) of the squared discrepency between 
        the predicted and measured states given a set of exponentiated rate parameters, by:
            Updating the rate parameters of the reaction system,
            Converting the reaction system into an ODE system (:class:`pyodesys.symbolic.SymbolicSys`),
            Solving the ODE system (to get the predicted states) based on the `initial_states` attribute,
            Extracting the normalised visible states from all those predicted
        '''

        # Update the rate params of the reaction system 
        self.reaction_system.update_rate_params( 2**rate_params_ex )

        # Convert to ODE system
        ode_system, _ = get_odesys( 
            self.reaction_system,
            # unit_registry=SI_base_registry,
            # output_conc_unit=( default_units.mass * 10e6 / ( (default_units.metre ** 3) * 1000 )),
            # output_time_unit=( default_units.second * 60 * 60 )
            )

        # Solve the ODE system (states predicted)
        states_p = ode_system.integrate(
            self.eval_times, # evaluation times
            self.states_0,  # initial states
            atol=self.atol, 
            rtol=self.rtol
        )

        # print( 'states_p', states_p )
        
        states_p = states_p.yout
        
        # Derive the Normalised Visible states from the Predicted states
        states_nvp = np.delete( states_p, self.ihs, 1 ) / self.max_measured
        # print( 'states_nvp', states_nvp )
        # print( 'states_nm', self.states_nm )

        del states_p

        # Calculate the net error: sum (over time) of the discrepencies squared
        discrepency = states_nvp[1:, :] - self.states_nm[1:, :]
        net_error = np.sum( np.multiply( discrepency, discrepency ) )

        return net_error 

#------------------------------------------------------------------------------------------

    def optimise( self, n_epochs=3, n_hops=10, display=True, atol=1e-12, rtol=1e-13 ):
        
        # Setup
        n = 0
        setattr( self, 'atol', atol )
        setattr( self, 'rtol', rtol )
        random_rates = lambda low, high: \
            np.random.uniform( low=low, high=high, size=self.num_rxns )
        
        fails = 0
        # Loop over epochs
        while n < n_epochs:
            
            if display:
                print(f'\nEpoch {n+1}:') 

            rate_param_ex = random_rates(-2, 1.5)

            try:
                rate_params_ex = basinhopping(
                    self.objective, 
                    rate_param_ex, 
                    minimizer_kwargs={'method': 'L-BFGS-B'},
                    # T=None,
                    niter=n_hops, 
                    disp=display, 
                    take_step=custom_hop,
                    callback=None
                ).x 
            except:
                print(' Basinhopping failed.\n')
                n -= 1
                fails += 1
                if fails > 5:
                    break
            finally:
                n += 1
            
        # Set optimal reaction rates
        setattr( self, 'optimal_rate_params', 2 ** rate_params_ex )

        # Get optimal prediction:
        self.reaction_system.update_rate_params( self.optimal_rate_params )
        ode_system, _ = get_odesys( self.reaction_system )
        states_p = ode_system.integrate(
            self.eval_times,  #self.eval_times, # evaluation times
            self.states_0,  # initial states
            atol=atol,  
            rtol=rtol
        )

        # Set predicted states attribute
        setattr( self, 'states_p', states_p )
        
        if display:
            print( 'Optimal predicted states\n', states_p)

        return rate_params_ex

#------------------------------------------------------------------------------------------

    def save_results( 
        self, 
        eval_times=None,
        ignore=None, 
        predicted=True, 
        measured=True,
        plot_name='',
        plot_name_stem='Leaching of Bauxite Residue:'
        # timestamp=False
        ): 

        # Update predicted states, if required
        if eval_times is None:
            # Use existing attributes
            eval_times = self.eval_times
            states_p = self.states_p
        else:
            # Given evaluation times, derive new prediction 
            ode_system, _ = get_odesys( self.reaction_system )
            states_p = ode_system.integrate(
                eval_times, 
                self.states_0, # initial states
                atol=self.atol,
                rtol=self.rtol
            )

        # Save results to .csv
        print( 'Predicted states', states_p.yout )
        np.savetxt('results/leaching/states/predicted_states.csv', states_p.yout, delimiter=',')
        np.savetxt('results/leaching/states/species.csv', self.species, fmt='%s', delimiter=',')
        np.savetxt('results/leaching/states/eval_times.csv', states_p.xout, delimiter=',')
        np.savetxt('results/leaching/kinetics/optimal_rate_params.csv', self.optimal_rate_params, delimiter=',')

        # Plot predicted and measured states
        if ignore is None:
            ignore = [ s for s in self.species if s not in self.species_m]
        colours = plt.cm.rainbow(np.linspace(0, 1, len(self.species)))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax in axes:
            # Plot predicted states
            for i, s in enumerate( self.species ):
                if s not in ignore:
                    if predicted:
                        ax.plot( 
                            eval_times,
                            states_p.yout[:, i],
                            linestyle='dashed',
                            label=s + ' (predicted)',
                            c=colours[i]
                    )
                    if measured and s in self.species_m:
                        j = self.species_m.index(s)
                        ax.plot( 
                        self.eval_times,
                        self.states_m[:, j],
                        linestyle = 'None',
                        marker='.',
                        ms=6,
                        label=s + ' (measured)',
                        c=colours[i]
                    )
                       
                        
                # # _ = states_p.plot( 
                # #     names=[k for k in self.reaction_system.substances if k not in ignore], 
                # #     ax=ax
                # #     )
                # if measured:
                # # Plot measured states
                #     for i, s in enumerate( self.species_m ):
                #         ax.plot( 
                #             self.eval_times, \
                #             self.states_m[:, i], \
                #             linestyle = 'None', \
                #             marker='.', \
                #             label=s + ' (measured)'
                #         )
            # Set legend and axes' lables
            _ = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

            _ = ax.set_xlabel('Time (days)')
            _ = ax.set_ylabel('Concentration (mg/L)')
        # Adjust titles and scales
        _ = axes[1].set_xscale('log')
        _ = axes[1].set_yscale('log')
        axes[0].set_title('Normal scale', loc='left')
        axes[1].set_title('Log scale', loc='left')
        
        # Tidy and Save
        _ = axes[0].legend().remove()
        plot_name_middle = ' predicted and measured concentrations over time '
        suptitle = plot_name_stem + plot_name_middle + plot_name 
        fig.suptitle( suptitle, fontsize=16 )
        _ = fig.tight_layout()
        save_at = 'results/leaching/plots/' + plot_name_stem + plot_name_middle + plot_name + '.png'
        _ = fig.savefig( save_at, dpi=72 )