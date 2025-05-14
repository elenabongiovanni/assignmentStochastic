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


def inSaampleStability_ATO(n_scenarios, num_variables_ATO, setting_ATO, tol):

    val_1 = 100
    val_2 = 0

    while abs(val_1-val_2) > tol:

        stoch_model_ATO_1 = StochModel(num_variables_ATO, setting_ATO)
        stoch_model_ATO_2 = StochModel(num_variables_ATO, setting_ATO)

        initial_value_ATO = setting_ATO.get('expectedValue')

        tree_ATO_1 = ScenarioTree(name="Tree ATO",branching_factors=[n_scenarios], len_vector=1, initial_value=initial_value_ATO, stoch_model=stoch_model_ATO_1)
        tree_ATO_2 = ScenarioTree(name="Tree ATO",branching_factors=[n_scenarios], len_vector=1, initial_value=initial_value_ATO, stoch_model=stoch_model_ATO_2)

        [prob_ATO_1,demand_ATO_1] = stoch_model_ATO_1.simulate_one_time_step(None, n_scenarios)
        [prob_ATO_2,demand_ATO_2] = stoch_model_ATO_2.simulate_one_time_step(None, n_scenarios)

        print('demand 1', demand_ATO_1)
        print('demand 2',demand_ATO_2)

        ottimo_1, val_1 = assembleToOrder(demand_ATO_1, prob_ATO_1, parameters_ATO)
        ottimo_2, val_2 = assembleToOrder(demand_ATO_2, prob_ATO_2, parameters_ATO)

        print('n_scenario', n_scenarios)
        n_scenarios = n_scenarios + 1
        print('differenza',abs(val_1-val_2))
        

    return n_scenarios


def inSampleStability_NewsVendor(n_scenario, num_variables_NewsVendor, setting_NewsVendor, tol):

    val_1 = 100
    val_2 = 0

    while abs(val_1-val_2) > tol:

        stoch_model_NewsVendor_1 = StochModel(num_variables_NewsVendor, setting_NewsVendor)
        stoch_model_NewsVendor_2 = StochModel(num_variables_NewsVendor, setting_NewsVendor)

        initial_value_NewsVendor = setting_NewsVendor.get('expectedValue')

        tree_NewsVendor_1 = ScenarioTree(name="Tree NewsVendor",branching_factors=[n_scenario], len_vector=1, initial_value=initial_value_NewsVendor, stoch_model=stoch_model_NewsVendor_1)
        tree_NewsVendor_2 = ScenarioTree(name="Tree NewsVendor",branching_factors=[n_scenario], len_vector=1, initial_value=initial_value_NewsVendor, stoch_model=stoch_model_NewsVendor_2)

        [prob_NV_1,demand_NV_1] = stoch_model_NewsVendor_1.simulate_one_time_step(None, n_scenario)
        [prob_NV_2,demand_NV_2] = stoch_model_NewsVendor_1.simulate_one_time_step(None, n_scenario)

        ottimo_1, val_1 = newsVendor(demand_NV_1[0], prob_NV_1, parameters_NewsVendor)
        ottimo_2, val_2 = newsVendor(demand_NV_2[0], prob_NV_2, parameters_NewsVendor)

        n_scenario = n_scenario + 1
        print('differenza',abs(val_1-val_2))


    return n_scenario











