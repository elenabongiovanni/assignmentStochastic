import numpy as np
from instances import *
from solvers import *
import json
import newsVendor
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist
from data import *
import solvers.scenarioTree as scenarioTree 

demand = generation.generate_random_demand(50, 0, 20)
prob = generation.generate_random_prob(50)
price = 10
cost = 3

# Parametri di esempio
branching_factors = [3, 3]  # 3 scenari al primo stadio, 3 al secondo
stoch_model = StochModel(1)
print(stoch_model.expected)
initial_value = [stoch_model.expected]  # valore iniziale della decisione (es. giornali acquistati)

# Crea l'albero degli scenari
tree = ScenarioTree(name="alberello",branching_factors=branching_factors, len_vector=1, initial_value=initial_value, stoch_model=stoch_model)

# Plotta l'albero
tree.plot()
#chiamo clustering

#chiamo newsvendor