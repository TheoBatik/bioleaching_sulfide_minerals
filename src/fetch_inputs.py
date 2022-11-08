import os 
import pandas as pd
from collections import defaultdict


def fetch_measured_states( file_name=None ):
    if file_name is not None:
        measured_states = pd.read_csv( os.path.join('data', 'clean', f'{file_name}.csv') )
    else:
        measured_states = [pd.read_csv( os.path.join('data', 'clean', f'dataset_{i}.csv') ) for i in range(1,4)]
    return measured_states


def fetch_initial_conditions( species, file_name='c0_base' ):
    
    # Setup
    c0_base = pd.read_csv( os.path.join('data', 'clean', f'{file_name}.csv') )
    base_species = c0_base.columns
    c0 = {}

    # Set c0 values
    for s in species: 
        if s in base_species:
            c0[s] = c0_base[s]
        else: 
            c0[s] = 0

    # Convert to default dictionary
    c0 = defaultdict(float, c0)

    return c0
