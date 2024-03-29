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
    def set_system( self ):
        '''
        Sets the reaction system (:class:`ReactionSystem`)
        for the hard-coded reactions (:class:`Reaction`)
        '''
    
        # Non-stoichiometic coefficient for Pyrrhotite
        x = 0 

        # Stoichiometric coefficients of reactants and products
        reactants = [
            # Chemical
            {'Pentlandite': 2, 'Fe3+': 36},
            {'Chalcopyrite': 1, 'Fe3+': 3},
            {'Chalcopyrite': 1, 'H+': 4, 'O': 2},
            {f'Pyrrhotite_{x}': 1, 'Fe3+': 8-2*x, 'H2O': 4},
            {'Pyrite': 1, 'Fe3+': 6, 'H2O': 3},
            {'(S2O3)2-': 1, 'Fe3+': 8, 'H2O': 5},
            # Microbial
            {'S': 8, 'O': 32 },
            {'Fe2+': 4, 'O': 2, 'H+': 4} # fix k for microbial reaction
        ]
        products = [
            # Chemical
            {'Ni2+': 9, 'Fe2+': 45, 'S': 16},
            {'Cu2+': 1, 'Fe2+': 5, 'S': 2},
            {'Cu2+': 1, 'Fe2+': 1, 'S': 2, 'H2O': 2},
            {'Fe2+': 9-3*x, '(SO4)2-': 1, 'H+': 8},
            {'(S2O3)2-': 1, 'Fe2+': 7, 'H+': 6},
            {'(SO4)2-': 1, 'Fe2+': 8, 'H+': 10},
            # Microbial
            {'(SO4)2-': 8},
            {'Fe3+': 4, 'H2O': 2}
        ]

        # INITIAL CONDITIONS! add acid [] based on pH, + Ferrous iron
        # Add Microbial reactions:
        # 1) ferrous to Feric  x 
        # 2) sulfur to sulfuric acid x 


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
            atol=1e-12,  
            rtol=1e-13
        ).yout
        
        # Derive the Normalised Visible states from the Predicted states
        states_nvp = np.delete( states_p, self.ihs, 1 ) / self.max_measured
        del states_p

        # Calculate the net error: sum (over time) of the discrepencies squared
        discrepency = states_nvp[1:, :] - self.states_nm[1:, :]
        net_error = np.sum( np.multiply( discrepency, discrepency ) )

        # print( 'net_error = ', net_error )
        return net_error 

#------------------------------------------------------------------------------------------

    # def generate_random_rate_param_ex( self ):
    #     '''Generate random exponentiated rate parameter'''
    #     rate_param_ex = \
    #         np.random.uniform( low=0.6, high=1.4, size=self.num_rxns ) + \
    #         np.random.uniform( low=0.1, high=0.2, size=self.num_rxns )
    #     return rate_param_ex

#------------------------------------------------------------------------------------------

    def optimise( self, n_epochs=3, n_hops=10, display=True, atol=1e-12, rtol=1e-13 ):
        
        # Setup
        random_rates = lambda low, high: \
            np.random.uniform( low=low, high=high, size=self.num_rxns )
        n = 0
        setattr( self, 'atol', atol )
        setattr( self, 'rtol', rtol )
        # Loop over epochs
        while n < n_epochs:
            
            if display:
                print(f'\nEpoch {n+1}:') 

            rate_param_ex = random_rates(-3, 2)
            # Generate random exponentiated rate parameter
            # rate_param_ex = np.concatenate(
            #     (random_rates(0.6, 1.4), random_rates(0.1, 0.2))
            #     )
            # rate_param_ex = random_rates(0.6, 1.4)
            # print(rate_param_ex)
            # try:
            rate_params_ex = basinhopping(
                self.objective, 
                rate_param_ex, 
                minimizer_kwargs={"method": "L-BFGS-B"}, 
                # T=None,
                niter=n_hops, 
                disp=display, 
                take_step=custom_hop,
                callback=None
            ).x 
            # except:
            #     pass
            # finally:
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
        
        # if display:
        #     print( 'Optimal predicted states\n', states_p)

        return rate_params_ex

#------------------------------------------------------------------------------------------

    def save_results( 
        self, 
        eval_times=None,
        ignore=None, 
        predicted=True, 
        measured=True,
        plot_name='',
        plot_name_stem='Leaching of Sulphide Minerals:'
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
        np.savetxt('results/states/SM_predicted_states.csv', states_p.yout, delimiter=',')
        np.savetxt('results/states/SM_species.csv', self.species, fmt='%s', delimiter=',')
        np.savetxt('results/states/SM_eval_times.csv', states_p.xout, delimiter=',')
        np.savetxt('results/kinetics/SM_optimal_rate_params.csv', self.optimal_rate_params, delimiter=',')

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
        # Adjust titles and scales[1].set_yscale('log')
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
        save_at = 'results/plots/' + plot_name_stem + plot_name_middle + plot_name + '.png'
        _ = fig.savefig( save_at, dpi=72 )




    def plot_total_Fe( 
        self, 
        eval_times=None,
        ignore=None, 
        predicted=True, 
        measured=True,
        plot_name='',
        plot_name_stem='Leaching of Sulphide Minerals:'
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

        # Plot predicted and measured states
        colours = plt.cm.rainbow(np.linspace(0, 1, len(self.species)))
        print( 'Adding', self.species[4], 'and', self.species[5])
        fep = states_p.yout[:, 4] + states_p.yout[:, 5]
        fem = self.states_m[:, 1] + self.states_m[:, 1]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax in axes:
        # Plot predicted states
            if predicted:
                ax.plot( 
                    eval_times,
                    fep,
                    linestyle='dashed',
                    label='Fe (predicted)',
                    c=colours[1]
            )
            if measured:
                ax.plot( 
                    self.eval_times,
                    fem,
                    linestyle = 'None',
                    marker='.',
                    ms=6,
                    label='Fe (measured)',
                    c=colours[3]
                )
                        
            # Set legend and axes' lables
            _ = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
            _ = ax.set_xlabel('Time (days)')
            _ = ax.set_ylabel('Concentration (mg/L)')
        # Adjust titles and scales[1].set_yscale('log')
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
        save_at = 'results/plots/' + plot_name_stem + plot_name_middle + plot_name + '.png'
        _ = fig.savefig( save_at, dpi=72 )