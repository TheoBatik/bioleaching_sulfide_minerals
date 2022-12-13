#------------------------------------------------------------------------------------------

# Imports
from chempy.chemistry import Reaction
from chempy import ReactionSystem
from chempy.kinetics.ode import get_odesys
import numpy as np

#------------------------------------------------------------------------------------------

# Extend the :class:`ReactionSystem` with a method update its own reaction rate params

def update_rate_param( reaction, rate_param ):
    'Updates the reaction rate `param` for single instance of :class:`Reaction`'
    setattr( reaction, 'param', rate_param )
    return reaction

class ReactionSystemExtended( ReactionSystem ):

    def update_rate_params( self, rate_params ):
        '''
        Update the reaction rate `param` of all :class:`Reaction`s
        in the :class:`ReactionSystem`
        '''

        # Update the rate params of each reaction
        reactions = list( map( update_rate_param, iter(self.rxns), iter(rate_params) ) )
    
        # Update the reactions of the reaction system
        setattr(self, 'rxns', reactions)

#------------------------------------------------------------------------------------------

class RateOptimiser:
    '''
    ADD DESCRIPTION 

    Parameters
    ----------
        states_measured: :pandas.DataFrame:
        states_initial: :pandas.DataFrame: 
    '''
#------------------------------------------------------------------------------------------

    def set_system( self, reversible=False ):
        '''
        Sets the reaction system (:class:`ReactionSystem`)
        for the hard-coded reactions (:class:`Reaction`)
        '''
    
        # Non-stoichiometic coefficient for Pyrrhotite
        x = 0 

        # Stoichiometric coefficients of reactants and products
        reactants = [ 
            {'Pentlandite': 2, 'Fe3+': 36},
            {'Chalcopyrite': 1, 'Fe3+': 3},
            {'Chalcopyrite': 1, 'H+': 4, 'O': 2},
            {f'Pyrrhotite_{x}': 1, 'Fe3+': 8-2*x, 'H2O': 4},
            {'Pyrite': 1, 'Fe3+': 6, 'H2O': 3},
            {'(S2O3)2-': 1, 'Fe3+': 8, 'H2O': 5}, 
            {'S': 8, 'O': 32 }
        ]
        products = [
            {'Ni2+': 9, 'Fe2+': 45, 'S': 16},
            {'Cu2+': 1, 'Fe2+': 5, 'S': 2},
            {'Cu2+': 1, 'Fe2+': 1, 'S': 2, 'H2O': 2},
            {'Fe2+': 9-3*x, '(SO4)2-': 1, 'H+': 8},
            {'(S2O3)2-': 1, 'Fe2+': 7, 'H+': 6},
            {'(S04)2-': 1, 'Fe2+': 8, 'H+': 10},
            {'(SO4)2-': 8}
        ]

        # INITIAL CONDITIONS! add acid [] based on pH, + Ferrous iron
        # Add Microbial reactions:
        # 1) ferrous to Feric
        # 2) sulfur to sulfuric acid


        # Number of reactions
        num_rxns = len( reactants ) 

        # Forward rate params & reactions
        forward_rate_params = np.random.uniform( low=0.5, high=1, size=num_rxns ) # forward reaction rate params
        forward_reactions = [ Reaction( r, p, k ) for r, p, k in zip( reactants, products, forward_rate_params ) ]
        reactions = forward_reactions 

        # Backward rate params & reactions
        if reversible:
            backward_rate_params = np.random.uniform( low=0.1, high=0.5, size=num_rxns ) # backward reaction rate params
            backward_reactions = [ Reaction( r, p, k ) for r, p, k in zip( products, reactants, backward_rate_params ) ]
            reactions += backward_reactions 
        
        # Set species
        species = set().union( *[ rxn.keys() for rxn in reactions ] )
        setattr( self, 'species', species)

        # Set reaction system
        reaction_system = ReactionSystemExtended( reactions, species )
        setattr( self, 'reaction_system', reaction_system )

#------------------------------------------------------------------------------------------

    def input( self, measured_states, initial_states ):

        # Set input attributes
        setattr( self, 'states_m', measured_states )
        setattr( self, 'states_0', initial_states )
        
        # Extract times at which to evalutate the solution of the ODE system during optimisation
        setattr( self, 'eval_times', list( measured_states['Time (hours)'] ))
        print(self.eval_times)


#------------------------------------------------------------------------------------------

    def objective( self, rate_params ):
        '''
        Updates the reaction `rate_params` of the reaction system
        Converts the reaction system into an ODE system (:class:`pyodesys.symbolic.SymbolicSys`)
        Solves the ODE system for the predicted states using the initial states attribute
        Calculates the error as sum over time of the squared discrepency between the predicted and measured states
        '''

        # Update rate params 
        self.reaction_system.update_rate_params( rate_params )
        
        # print(self.states_0)
        # print(self.eval_times)

        # Convert to ODE system
        ode_system, _ = get_odesys( self.reaction_system )

        # Solve ODE system for states predicted
        states_p = ode_system.integrate(
            self.eval_times, # evaluation times
            self.states_0,  # initial states
            atol=1e-12,  
            rtol= 1e-13
        )
        
        # print( ' states_0.xout ', self.states_0)

        # for key in self.states_0.keys():
        #     print(key, self.states_0[key])

        print( ' states_p[0].xout ', states_p.xout)
        print( ' states_p[0].yout ', states_p.yout)

        # Plot
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax in axes:
            plt.plot(states_p.xout, states_p.yout )
            plt.plot( self.eval_times, self.states_m[ 'Cu' ] )
            plt.plot( self.eval_times, self.states_m[ 'Ni' ] )
            plt.plot( self.eval_times, self.states_m[ 'Fe' ] )

            # _ = states_p[0].plot(names=[k for k in self.reaction_system.substances if k != 'H2O'], ax=ax)
            _ = ax.legend(loc='best', prop={'size': 9})
            _ = ax.set_xlabel('Time')
            _ = ax.set_ylabel('Concentration')
        # _ = axes[1].set_ylim([1e-13, 1e-1])
        _ = axes[1].set_xscale('log')
        _ = axes[1].set_yscale('log')
        _ = fig.tight_layout()
        _ = fig.savefig('test_objective.png', dpi=72)



        return states_p


#------------------------------------------------------------------------------------------


# def objective( ode_system ):
    # # Delete hidden states from the full set predicted
    # predicted_visible_states = np.delete( self.evolve_network( k_expo ), self.hidden_states, 0 )

    # # Normalise the visible states predicted (using the measured states' maxes)
    # predicted_visible_states_norm = predicted_visible_states.T / self.column_maxes
        
    # # Compute the sum over time of the squared discrepency between the predicted and measured states
    # discrepency = predicted_visible_states_norm[:, :] - self.measured_visible_states_norm[:, :]
    # error_squared = np.sum( np.multiply( discrepency, discrepency ) )

    # state_predicted = ode_system.integrate( t_out, c0, atol=1e-12, rtol=1e-14 )
    # pass

# 