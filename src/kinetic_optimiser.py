# Imports
from chempy.kinetics.ode import get_odesys


class KineticOptimiser:
    '''
    DESCRIPTION 

    Parameters
    ----------
        reaction_system: :class:`ReactionSystem`
        states_measured: :pandas.DataFrame:
        states_initial: :pandas.DataFrame:
        
    '''
    def __init__(
        self, 
        reaction_system, 
        states_measured,
        states_initial
    ):
        
        # Set key attributes
        setattr( self, 'reaction_system', reaction_system )
        setattr( self, 'states_measured', states_measured )
        setattr( self, 'states_initial', states_initial )
        
        # Time points at which to evalutate the solution of the ODE system
        self.t_out = self.states_measured['Time (hours)']

        # self.states_hidden = 
        # setattr( self, 'output_folder'. )


    def objective( self ):
        '''
        Reaction System --> ODE System:
        The :meth: rateExp() of each :class:`Reaction` 
        in the :class:`ReactionSystem` is invoked  
        '''

        # Set ODE system, updating each reactions rate
        ode_system, _ = get_odesys( self.reaction_system )
        
        # Solve ODE system
        states_predicted = ode_system.integrate(
            self.t_out, 
            self.states_initial, 
            atol=1e-12, 
            rtol=1e-14
        )

        return states_predicted



# def objective( ode_system ):
    # # Delete hidden states from the full set predicted
    # predicted_visible_states = np.delete( self.evolve_network( k_expo ), self.hidden_states, 0 )

    # # Normalise the visible states predicted (using the measured states' maxes)
    # predicted_visible_states_norm = predicted_visible_states.T / self.column_maxes
        
    # # Compute the sum over time of the squared discrepency between the predicted and measured states
    # discrepency = predicted_visible_states_norm[:, :] - self.measured_visible_states_norm[:, :]
    # error_squared = np.sum( np.multiply( discrepency, discrepency ) )

    # state_predicted = ode_system.integrate( t_out, c0, atol=1e-12, rtol=1e-14 )
    pass


def update_reaction_rate( reaction, rate ):
    setattr()


def update_ode_system_rates( reaction_system, rates ):
    '''
    Updates:
        The rate parameter of each :class:`Reaction` 
        in the :class:`ReactionSystem.`

    Returns:
        The associated :class:`ode_system`
    '''
   
    # Update rate function
    update_rate = lambda rxn, rate: setattr( rxn, 'param', rate )
    reactions = list( map( update_rate, iter(reaction_system.rxns), iter(rates) ) )

    # Update reaction system
    setattr(reaction_system, 'rxns', reactions)

    # Convert to ODE system
    ode_system, _ = get_odesys( reaction_system )

    return ode_system