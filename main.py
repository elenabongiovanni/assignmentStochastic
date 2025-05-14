import numpy as np

from clustering.wasserstainDistance import *
from data.parameters import *
from problems.assembleToOrder import *
from problems.newsVendor import *
from instances import *
from result import *
from setting.scenarioSetting import *
from solvers import *

from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist

#### MODELLO ####

branching_factors = [12]  

num_variables_ATO = parameters_ATO.get('n_items')
num_variables_NewsVendor = 1

setting_ATO = scenario_setting_ATO
setting_NewsVendor = scenario_setting_NewsVendor

stoch_model_ATO = StochModel(num_variables_ATO, setting_ATO)
stoch_model_NewsVendor = StochModel(num_variables_NewsVendor, setting_NewsVendor)

initial_value_ATO = setting_ATO.get('expectedValue')
initial_value_NewsVendor = setting_NewsVendor.get('expectedValue')

# Crea l'albero degli scenari
tree_ATO = ScenarioTree(name="Tree ATO",branching_factors=branching_factors, len_vector=1, initial_value=initial_value_ATO, stoch_model=stoch_model_ATO)
#tree_ATO.plot()

tree_NewsVendor = ScenarioTree(name="Tree NewsVendor",branching_factors=branching_factors, len_vector=1, initial_value=initial_value_NewsVendor, stoch_model=stoch_model_NewsVendor)
#tree_NewsVendor.plot()


# Salvo gli scenari
scenari_ATO = tree_ATO.get_all_scenarios()
scenari_NewsVendor = tree_NewsVendor.get_all_scenarios()
#best_subset, scenari[best_subset], best_distance = reduce_scenarios_nearest_neighbors(scenari, 20, p=2)
#print(best_subset)
#print(scenari[best_subset])


#### ATO ####

#domanda2=np.array([[100, 50, 120],[50,25,60],[100,110,60]])
#prob2= np.array([1/3,1/3,1/3])
[prob_ATO,demand_ATO] = stoch_model_ATO.simulate_one_time_step(None, branching_factors[0])
parameters_ato = parameters_ATO 

ottimo2, val2 = assembleToOrder(demand_ATO, prob_ATO, parameters_ato)


#### NEWSVEDOR ####

#domanda3 = [0,1,2,3]
#prob3 = [0.4, 0.3, 0.2, 0.1]
parameters_newsvendor = parameters_NewsVendor
[prob_NV,demand_NV] = stoch_model_NewsVendor.simulate_one_time_step(None, branching_factors[0])

ottimo3, val3 = newsVendor(demand_NV[0], prob_NV, parameters_newsvendor)