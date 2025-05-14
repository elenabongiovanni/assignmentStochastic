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

# per verificare quanto la soluzione sia generalizzabile, ovvero se funziona bene anche su scenari mai visti prima

def evaluate_solution(solution, demand, probabilities, parameters, objective_function):
    """
    Valuta il valore atteso di una soluzione fissa data una nuova domanda e probabilità.
    """
    total_value = 0
    for i in range(len(demand)):
        total_value += probabilities[i] * objective_function(solution, demand[i], parameters)
    return total_value


def outOfSampleStabilityATO(n_scenarios_train, n_scenarios_test, num_variables_ATO, setting_ATO, n_repeat = 5):
    # a differenza della in sample devo ripetere il test più volte per avere una stima affidabile della performance

    # genero gli scenari di training
    stoch_model_ATO_train  = StochModel(num_variables_ATO, setting_ATO)
    initial_value_ATO = setting_ATO.get('expectedValue')
    #tree_ATO_train = ScenarioTree(name="Tree ATO Train", branching_factors=[n_scenarios], len_vector=1, initial_value=initial_value_ATO, stoch_model=stoch_model_ATO_train)
    [prob_ATO_train,demand_ATO_train] = stoch_model_ATO_train.simulate_one_time_step(None, n_scenarios_train)
    ottimo_train, val_train = assembleToOrder(demand_ATO_train, prob_ATO_train, parameters_ATO)

    out_of_sample_values = []

    for i in range(num_repeat):
        stoch_model_ATO_test = StochModel(num_variables_ATO, setting_ATO)
        [prob_ATO_test,demand_ATO_test] = stoch_model_ATO_test.simulate_one_time_step(None, n_scenarios_test)

        # Valuta la soluzione trovata sui nuovi scenari
        value_test = evaluate_solution(ottimo_train, demand_ATO_test, prob_ATO_test, parameters_ATO, compute_objective_ATO)
        out_of_sample_values.append(value_test)

    std_dev = np.std(out_of_sample_values)
    
    return std_dev



def outOfSampleStability_NewsVendor(n_scenarios_train, n_scenarios_test, num_variables_NV, setting_NV, n_repeat = 5):

    # genero gli scenari di training
    stoch_model_NV_train  = StochModel(num_variables_NV, setting_NV)
    initial_value_NV = setting_NV.get('expectedValue')
    #tree_ATO_train = ScenarioTree(name="Tree ATO Train", branching_factors=[n_scenarios], len_vector=1, initial_value=initial_value_ATO, stoch_model=stoch_model_ATO_train)
    [prob_NV_train,demand_NV_train] = stoch_model_NV_train.simulate_one_time_step(None, n_scenarios_train)
    ottimo_train, val_train = newsVendor(demand_NV_train, prob_NV_train, parameters_NV)

    out_of_sample_values = []

    for i in range(num_repeat):
        stoch_model_NV_test = StochModel(num_variables_NV, setting_NV)
        [prob_ATO_test,demand_ATO_test] = stoch_model_NV_test.simulate_one_time_step(None, n_scenarios_test)

        # Valuta la soluzione trovata sui nuovi scenari
        value_test = evaluate_solution(ottimo_train, demand_NV_test, prob_NV_test, parameters_NV, compute_objective_NewsVendor)
        out_of_sample_values.append(value_test)

    std_dev = np.std(out_of_sample_values)
    
    return std_dev


    