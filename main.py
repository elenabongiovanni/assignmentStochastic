import numpy as np

from clustering.wasserstainDistance import *
from clustering.momentMatching import *
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

branching_factors = [200]  

###########################
######## NEWSVENDOR ########
###########################
print("Newsvendor")
##### SIMULATION WITH ROUGH MC #####

num_variables_NewsVendor = 1
setting_NewsVendor = scenario_setting_NewsVendor

stoch_model_NewsVendor = StochModel(num_variables_NewsVendor, setting_NewsVendor)

initial_value_NewsVendor = setting_NewsVendor.get('expectedValue')

# Create ScenarioTree
tree_NewsVendor = ScenarioTree(name="Tree NewsVendor",branching_factors=branching_factors, len_vector=1, initial_value=initial_value_NewsVendor, stoch_model=stoch_model_NewsVendor)
#tree_NewsVendor.plot()

# Save Scenarios
scenari_NewsVendor = tree_NewsVendor.get_all_scenarios()
prob_Newsvendor = [float(tree_NewsVendor.nodes[leaf]['path_prob']) for leaf in tree_NewsVendor.leaves] 
#print(scenari_NewsVendor)
#print(prob_Newsvendor)
# Solve Newsvendor problem using generated scenarios
#domanda3 = [0,1,2,3]
#prob3 = [0.4, 0.3, 0.2, 0.1]
parameters_newsvendor = parameters_NewsVendor
#[prob_NV,demand_NV] = stoch_model_NewsVendor.simulate_one_time_step(None, branching_factors[0])
#print(prob_NV)
ottimo_NV, val_NV = newsVendor(scenari_NewsVendor.reshape(-1), prob_Newsvendor, parameters_newsvendor)

#### SIMULATION WITH REDUCED SCENARIOS: MOMENT MATCHING
num_reduce = 10

sigma2 = scenario_setting_NewsVendor.get('devstd')
mu = scenario_setting_NewsVendor.get('expectedValue')
weight = 1
NV_best_subset_MM, NV_best_distance_MM, NV_prob_MM = reduce_scenarios_momentmatching(scenari_NewsVendor, num_reduce,mu,sigma2,weight)
print("Best distance with Moment Matching: ",NV_best_distance_MM)
print("prob Moment Matching NV: ", NV_prob_MM)
#print(NV_best_subset_MM)

ottimo_NV_MM, val_NV_MM = newsVendor(NV_best_subset_MM[0], NV_prob_MM, parameters_newsvendor)

##### SIMULATION WITH REDUCED SCENARIOS: WASSERSTEIN DISTANCE

num_reduce = 20
NV_best_subset_W, NV_best_distance_W, NV_prob_W = reduce_scenarios_wasserstein(scenari_NewsVendor, num_reduce, p=2)
print("Best distance with Wasserstein: ",NV_best_distance_W)
ottimo_NV_W, val_NV_W = newsVendor(NV_best_subset_W.T[0], NV_prob_W, parameters_newsvendor)




###########################
########### ATO ###########
###########################
print("ATO")
##### SIMULATION WITH ROUGH MC #####

num_variables_ATO = parameters_ATO.get('n_items')
setting_ATO = scenario_setting_ATO

stoch_model_ATO = StochModel(num_variables_ATO, setting_ATO)

initial_value_ATO = setting_ATO.get('expectedValue')

# Create ScenarioTree
tree_ATO = ScenarioTree(name="Tree ATO",branching_factors=branching_factors, len_vector=1, initial_value=initial_value_ATO, stoch_model=stoch_model_ATO)
#tree_ATO.plot()

# Save scenarios
scenari_ATO = tree_ATO.get_all_scenarios()
prob_ATO = [float(tree_ATO.nodes[leaf]['path_prob']) for leaf in tree_ATO.leaves] 
#best_subset, scenari[best_subset], best_distance = reduce_scenarios_nearest_neighbors(scenari, 20, p=2)
#print(best_subset)
#print(scenari[best_subset])

#domanda2=np.array([[100, 50, 120],[50,25,60],[100,110,60]])
#prob2= np.array([1/3,1/3,1/3])

# Solve ATO problem using generated scenarios
#[prob_ATO,demand_ATO] = stoch_model_ATO.simulate_one_time_step(None, branching_factors[0])

parameters_ato = parameters_ATO 

#ottimo2, val2 = assembleToOrder(demand_ATO, prob_ATO, parameters_ato)
ottimo2, val2 = assembleToOrder(scenari_ATO.T, prob_ATO, parameters_ato)


#### SIMULATION WITH REDUCED SCENARIOS: MOMENT MATCHING
num_reduce = 20

sigma2 = scenario_setting_ATO.get('devstd')
mu = scenario_setting_ATO.get('expectedValue')
weight = 1
ATO_best_subset_MM, ATO_best_distance_MM, ATO_prob_MM = reduce_scenarios_momentmatching(scenari_ATO, num_reduce,mu,sigma2,weight)
print("Best distance with moment matching: ",ATO_best_distance_MM)
print("prob Moment Matching ATO: ", ATO_prob_MM)
#print(ATO_best_subset_MM)



ottimo_ATO_MM, val_ATO_MM = assembleToOrder(ATO_best_subset_MM, ATO_prob_MM, parameters_ato)


##### SIMULATION WITH REDUCED SCENARIOS: WASSERSTEIN DISTANCE
num_reduce = 20

ATO_best_subset_W, ATO_best_distance_W, ATO_prob_W = reduce_scenarios_wasserstein(scenari_ATO, num_reduce, p=2)
print("Best distance with Wasserstein: ",ATO_best_distance_W)
ottimo_ATO_W, val_ATO_W = assembleToOrder(ATO_best_subset_W.T, ATO_prob_W, parameters_ato)


