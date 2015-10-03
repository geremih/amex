
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt


df = pd.read_csv('training.csv')

df.columns = ['Citizen_ID', 'actual_vote', 'previous_vote', 'd_centaur', 'd_ebony', 'd_tokugawa', 'd_odyssey', 'd_cosmos', 's_centaur', 's_ebony', 's_tokugawa', 's_cosmos', 's_odyssey', 'occ', 'region', 'h_size', 'age', 'married', 'home', 'politics', 'r_years', 'p_voted', 'n_unique_p', 'education', 'n_centaur', 'n_ebony', 'n_tokugawa', 'n_odyssey', 'n_rallies', 'n_cosmos', 'docs', 'income']


df.loc[:,['d_centaur', 'd_ebony', 'd_tokugawa', 'd_odyssey', 'd_cosmos', 's_centaur', 's_ebony', 's_tokugawa', 's_cosmos', 's_odyssey', 'n_centaur', 'n_ebony', 'n_tokugawa', 'n_odyssey', 'n_rallies', 'n_cosmos']].fillna(0)



 



