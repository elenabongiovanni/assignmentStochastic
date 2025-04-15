import numpy as np
from instances import *
from solvers import *
import json
import newsVendor
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist
from data import *
import solvers.scenarioTree as scenarioTree 
import plantLocation
import newsVendor
import assembleToOrder

demand = generation.generate_random_demand(50, 0, 20)
prob = generation.generate_random_prob(50)
price = 10
cost = 3

class Scenario:
    def __init__(self, demand, prob):
        self.demand = demand
        self.prob = prob

# Parametri
n_production_nodes = 5
n_demand_nodes = 7
n_scenarios = 4

np.random.seed(42)

# Domanda stocastica per scenario
demand = np.random.randint(5, 20, size=(n_demand_nodes, n_scenarios))

# Probabilità degli scenari (somma = 1)
prob = np.random.dirichlet(np.ones(n_scenarios), size=1)[0]

# Costi di trasporto
transportation_cost = np.random.randint(1, 10, size=(n_production_nodes, n_demand_nodes))

# Costi fissi di apertura degli impianti
fixed_cost = np.random.randint(100, 500, size=n_production_nodes)

# Test funzione
ottimo, valore = plantLocation.plantLocation(demand, prob, transportation_cost, fixed_cost)

costi2=np.array([20,30,10,10,10])
domanda2=np.array([[100, 50, 120],[50,25,60],[100,110,60]])
salling_price2=np.array([80,70,90])
prob2= np.array([1/3,1/3,1/3])
ottimo2, val2 = assembleToOrder.assembleToOrder(domanda2, prob2, salling_price2, costi2)

"""
# Parametri di esempio
branching_factors = [3]  # 1 scenario al primo stadio, 3 al secondo stadio
stoch_model = StochModel(1)
print(stoch_model.expected)
initial_value = [stoch_model.expected]  # valore iniziale della decisione (es. giornali acquistati)

# Crea l'albero degli scenari
tree = ScenarioTree(name="alberello",branching_factors=branching_factors, len_vector=1, initial_value=initial_value, stoch_model=stoch_model)

# Plotta l'albero
tree.plot()
#chiamo clustering

#chiamo newsvendor
"""