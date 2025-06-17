import numpy as np

from clustering.wasserstainDistance import *
from clustering.momentMatching import *
from data.parameters import *
from problems.assembleToOrder import *
from problems.newsVendor import *
from result.results_NV import *
from result.results_ATO import *
from setting.scenarioSetting import *
from solvers import *

from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist

branching_factors = [100]  

############################
######## NEWSVENDOR ########
############################

print("Solving Newsvendor problem")

##### SIMULATION WITH ROUGH MC #####

num_variables_NewsVendor = 1
setting_NewsVendor = scenario_setting_NewsVendor
initial_value_NewsVendor = setting_NewsVendor.get('expectedValue')

stoch_model_NewsVendor = StochModel(num_variables_NewsVendor, setting_NewsVendor)

tree_NewsVendor = ScenarioTree(name="Tree NewsVendor",branching_factors=branching_factors, len_vector=1, initial_value=initial_value_NewsVendor, stoch_model=stoch_model_NewsVendor)
tree_NewsVendor.plot()
scenari_NewsVendor = tree_NewsVendor.get_all_scenarios()
prob_Newsvendor = [float(tree_NewsVendor.nodes[leaf]['path_prob']) for leaf in tree_NewsVendor.leaves] 

parameters_newsvendor = parameters_NewsVendor
ottimo_NV, val_NV = newsVendor(scenari_NewsVendor.reshape(-1), prob_Newsvendor, parameters_newsvendor)

#### SIMULATION WITH REDUCED SCENARIOS: MOMENT MATCHING

num_reduce = 50

sigma2 = scenario_setting_NewsVendor.get('devstd')
mu = scenario_setting_NewsVendor.get('expectedValue')
weight = 1

NV_best_subset_MM, NV_best_distance_MM, NV_prob_MM = reduce_scenarios_momentmatching(scenari_NewsVendor, num_reduce,mu,sigma2,weight)

ottimo_NV_MM, val_NV_MM = newsVendor(NV_best_subset_MM[0], NV_prob_MM, parameters_newsvendor)

##### SIMULATION WITH REDUCED SCENARIOS: WASSERSTEIN DISTANCE

num_reduce = 50

NV_best_subset_W, NV_best_distance_W, NV_prob_W = reduce_scenarios_wasserstein(scenari_NewsVendor, num_reduce, p=2)

ottimo_NV_W, val_NV_W = newsVendor(NV_best_subset_W.T[0], NV_prob_W, parameters_newsvendor)

save_newsvendor_results(val_NV, ottimo_NV, scenari_NewsVendor.reshape(-1), val_NV_MM, ottimo_NV_MM, NV_best_distance_MM, NV_best_subset_MM[0], val_NV_W,  ottimo_NV_W, NV_best_distance_W, NV_best_subset_W.T[0])


###########################
########### ATO ###########
###########################

print("Solving Assemble To Order problem")


##### SIMULATION WITH ROUGH MC #####

num_variables_ATO = parameters_ATO.get('n_items')
setting_ATO = scenario_setting_ATO
initial_value_ATO = setting_ATO.get('expectedValue')

stoch_model_ATO = StochModel(num_variables_ATO, setting_ATO)

tree_ATO = ScenarioTree(name="Tree ATO",branching_factors=branching_factors, len_vector=1, initial_value=initial_value_ATO, stoch_model=stoch_model_ATO)
tree_ATO.plot()
scenari_ATO = tree_ATO.get_all_scenarios()
prob_ATO = [float(tree_ATO.nodes[leaf]['path_prob']) for leaf in tree_ATO.leaves] 

parameters_ato = parameters_ATO 
ottimo_ATO, val_ATO = assembleToOrder(scenari_ATO.T, prob_ATO, parameters_ato)


#### SIMULATION WITH REDUCED SCENARIOS: MOMENT MATCHING

num_reduce = 50

mu = scenario_setting_ATO['expectedValue']
sigma = scenario_setting_ATO['devstd']
d = scenari_ATO.shape[1]
mu = np.full(d, mu[0])  
sigma2 = np.full(d, sigma**2)  

weight = 1

ATO_best_subset_MM, ATO_best_distance_MM, ATO_prob_MM = reduce_scenarios_momentmatching(scenari_ATO, num_reduce,mu,sigma2,weight)

ottimo_ATO_MM, val_ATO_MM = assembleToOrder(ATO_best_subset_MM, ATO_prob_MM, parameters_ato)


##### SIMULATION WITH REDUCED SCENARIOS: WASSERSTEIN DISTANCE

num_reduce = 50

ATO_best_subset_W, ATO_best_distance_W, ATO_prob_W = reduce_scenarios_wasserstein(scenari_ATO, num_reduce, p=2)

ottimo_ATO_W, val_ATO_W = assembleToOrder(ATO_best_subset_W.T, ATO_prob_W, parameters_ato)

save_ato_results(val_ATO, ottimo_ATO, scenari_ATO.T, val_ATO_MM, ottimo_ATO_MM, ATO_best_distance_MM, ATO_best_subset_MM.T, val_ATO_W, ottimo_ATO_W, ATO_best_distance_W, ATO_best_subset_W)


