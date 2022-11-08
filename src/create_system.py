from chempy.chemistry import Reaction
from chempy import ReactionSystem
import numpy as np


def create_reaction_system( reversible=False ): # add initial_rates
    '''
    Returns the reaction system (:class:`ReactionSystem`)
    for the underlying reactions (:class:`Reaction`)
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
        {'S': 8, 'O': 36 }, {'(SO4)2-': 8}
    ]
    products = [
        {'Ni2+': 9, 'Fe2+': 45, 'S': 16},
        {'Cu2+': 1, 'Fe2+': 5, 'S': 2},
        {'Cu2+': 1, 'Fe2+': 1, 'S': 2, 'H2O': 2},
        {'Fe2+': 9-3*x, '(SO4)2-': 1, 'H+': 8},
        {'(S2O3)2-': 1, 'Fe2+': 7, 'H+': 6},
        {'(S04)2-': 1, 'Fe2+': 8, 'H+': 10}
    ]

    # Number of reactions
    num_rxns = len(reactants) 

    # Forward rates & reactions
    forward_rates = np.random.uniform( low=0.5, high=1, size=num_rxns) # forward reaction rates
    forward_reactions = [ Reaction(r, p, k) for r, p, k in zip(reactants, products, forward_rates) ]
    reactions = forward_reactions 

    # Backward rates & reactions
    if reversible:
        backward_rates = np.random.uniform( low=0.1, high=0.5, size=num_rxns) # backward reaction rates
        backward_reactions = [ Reaction(r, p, k) for r, p, k in zip(products, reactants, backward_rates) ]
        reactions += backward_reactions 
    
    # Species
    species = set().union( *[rxn.keys() for rxn in reactions] )
    num_species = len(species)

    # Extra 
    # extra = {'num_rxns': num_rxns, 'species': species, 'num_species': num_species}

    return ReactionSystem(reactions, species), species
