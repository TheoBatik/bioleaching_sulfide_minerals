import os 
import pandas as pd
from collections import defaultdict
import numpy as np


def load_csv( csv_name='dataset_1' ):
    matrix = []
    with open( os.path.join('data', 'clean', f'{csv_name}.csv'), 'r', encoding='utf-8-sig') as file:
        for i, line in enumerate(file):
            if i == 0:
                header = list( line.strip().split(',') )
                continue
            row = [float(item) for item in line.strip().split(',')]
            matrix.append(row)
    matrix = np.asarray( matrix )
    matrix.astype( np.float32 )
    return matrix, header


def fetch_initial_states( species, csv_name='c0_base' ):
    '''     
    Returns :dict: c0
    The initial concetrations of all species
    based on the values in the csv.
    '''

    # Setup
    c0_base, base_species = load_csv( csv_name )
    c0 = {}

    # Set c0 values
    for s in species: 
        if s in base_species:
            index = base_species.index( s )
            c0[s] = float( c0_base[ 0, : ][ index ] )
        else: # c0 set to zero if species is not in file header
            c0[s] = float( 0 )


    return defaultdict( np.float32, c0 )


