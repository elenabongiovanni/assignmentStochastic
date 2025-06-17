import numpy as np

from clustering.wasserstainDistance import *
from data.parameters import *
from problems.assembleToOrder import *
from problems.newsVendor import *
from result import *
from setting.scenarioSetting import *
from solvers import *

from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist
from scipy.stats import norm

# Setting parameters
setting_ATO = scenario_setting_ATO
setting_NewsVendor = scenario_setting_NewsVendor

# ATO
def inSampleStability_ATO(n_scenarios_init, num_variables_ATO, setting_ATO, parameters_ATO, 
                          N, alpha, max_iter, tol):

    z_alpha = norm.ppf(1 - alpha / 2)  # Normal standard quantile implemented to create the confidence interval
    n_scenarios = n_scenarios_init
    iter_count = 0

    list_result = [] # Used to save the results of the stability

    while iter_count < max_iter:

        phi_list = []

        for i in range(N):
            
            # Generation of S and S' with cardinality n_scenarios 
            stoch_model_1 = StochModel(num_variables_ATO, setting_ATO)
            stoch_model_2 = StochModel(num_variables_ATO, setting_ATO)

            initial_value = setting_ATO.get('expectedValue')
            tree_ATO_1 = ScenarioTree("Tree ATO", [n_scenarios], 1, initial_value, stoch_model_1)
            tree_ATO_2 = ScenarioTree("Tree ATO", [n_scenarios], 1, initial_value, stoch_model_2)

            prob_1, demand_1 = stoch_model_1.simulate_one_time_step(None, n_scenarios)
            prob_2, demand_2 = stoch_model_2.simulate_one_time_step(None, n_scenarios)

            ottimo_1, val_1 = assembleToOrder(demand_1, prob_1, parameters_ATO)
            ottimo_2, val_2 = assembleToOrder(demand_2, prob_2, parameters_ATO)

            phi_list.append(val_1-val_2)

        phi_array = np.array(phi_list)
        mu = np.mean(phi_array)
        sigma = np.std(phi_array, ddof=1) 

        margin = z_alpha * sigma / np.sqrt(N)
        ci_lower = mu - margin
        ci_upper = mu + margin

        list_result.append({
                'n_scenarios': n_scenarios,
                'mu': mu,
                'devstd' : sigma,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
        })

        # Stopping conditionds: confidence interval and values differences
        if ci_lower <= 0 <= ci_upper and abs(val_1 - val_2) < tol:
            break

        # Update for next iteration
        n_scenarios += 20
        iter_count += 1

    return n_scenarios, list_result

# Newsvendor
def inSampleStability_NewsVendor(n_scenarios_init, num_variables_NewsVendor, setting_NewsVendor, 
                                  parameters_NewsVendor, N, alpha, max_iter, tol):

    z_alpha = norm.ppf(1 - alpha / 2) # Normal standard quantile implemented to create the confidence interval
    n_scenario = n_scenarios_init
    iter_count = 0

    list_result = [] # Used to save the results of the stability

    while iter_count < max_iter:
        phi_list = []

        for _ in range(N):

            # Generation of S and S' with cardinality n_scenario 
            stoch_model_1 = StochModel(num_variables_NewsVendor, setting_NewsVendor)
            stoch_model_2 = StochModel(num_variables_NewsVendor, setting_NewsVendor)

            initial_value = setting_NewsVendor.get('expectedValue')

            tree_1 = ScenarioTree("Tree NV", [n_scenario], 1, initial_value, stoch_model_1)
            tree_2 = ScenarioTree("Tree NV", [n_scenario], 1, initial_value, stoch_model_2)

            prob_1, demand_1 = stoch_model_1.simulate_one_time_step(None, n_scenario)
            prob_2, demand_2 = stoch_model_2.simulate_one_time_step(None, n_scenario)

            ottimo_1, val_1 = newsVendor(demand_1[0], prob_1, parameters_NewsVendor)
            ottimo_2, val_2 = newsVendor(demand_2[0], prob_2, parameters_NewsVendor)

            phi_list.append(val_1 - val_2)

        phi_array = np.array(phi_list)
        mu = np.mean(phi_array)
        sigma = np.std(phi_array, ddof=1)

        margin = z_alpha * sigma / np.sqrt(N)
        ci_lower = mu - margin
        ci_upper = mu + margin

        list_result.append({
                'n_scenario': n_scenario,
                'mu': mu,
                'devstd' : sigma,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
        })

        # Stopping conditionds: confidence interval and values differences
        if ci_lower <= 0 <= ci_upper and abs(val_1 - val_2) < tol:
            break
        
        # Update for next iteration
        n_scenario += 10
        iter_count += 1

    return n_scenario, list_result











