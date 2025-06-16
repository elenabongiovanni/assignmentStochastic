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

# per verificare che il numero di scenari utilizzato sia sufficiente per garantire una soluzione stabile, che non dipende troppo dal particolare campione di scenari

setting_ATO = scenario_setting_ATO
setting_NewsVendor = scenario_setting_NewsVendor

def inSampleStability_ATO(n_scenarios_init, num_variables_ATO, setting_ATO, parameters_ATO, 
                          N, alpha, max_iter, tol):

    z_alpha = norm.ppf(1 - alpha / 2)  # quantile normale standard
    n_scenarios = n_scenarios_init
    iter_count = 0

    list_result = []

    while iter_count < max_iter:

        phi_list = []

        for i in range(N):
            # Nuovi modelli stocastici indipendenti
            stoch_model_1 = StochModel(num_variables_ATO, setting_ATO)
            stoch_model_2 = StochModel(num_variables_ATO, setting_ATO)

            # Inizializzazione alberi di scenari indipendenti
            initial_value = setting_ATO.get('expectedValue')
            tree_ATO_1 = ScenarioTree("Tree ATO", [n_scenarios], 1, initial_value, stoch_model_1)
            tree_ATO_2 = ScenarioTree("Tree ATO", [n_scenarios], 1, initial_value, stoch_model_2)

            # Simulazione dei due alberi
            prob_1, demand_1 = stoch_model_1.simulate_one_time_step(None, n_scenarios)
            prob_2, demand_2 = stoch_model_2.simulate_one_time_step(None, n_scenarios)

            # Risoluzione del problema ATO
            ottimo_1, val_1 = assembleToOrder(demand_1, prob_1, parameters_ATO)
            ottimo_2, val_2 = assembleToOrder(demand_2, prob_2, parameters_ATO)

            phi_list.append(val_1-val_2)

        phi_array = np.array(phi_list)
        mu = np.mean(phi_array)
        sigma = np.std(phi_array, ddof=1)  # deviazione standard campionaria

        margin = z_alpha * sigma / np.sqrt(N)
        ci_lower = mu - margin
        ci_upper = mu + margin

        print(f"n_scenarios = {n_scenarios} | Intervallo di confidenza: [{ci_lower:.4f}, {ci_upper:.4f}]")

        list_result.append({
                'n_scenarios': n_scenarios,
                'mu': mu,
                'devstd' : sigma,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
        })

        # Condizione di arresto: 0 dentro l'intervallo di confidenza
        if ci_lower <= 0 <= ci_upper and abs(val_1 - val_2) < tol:
            print("Condizione di stabilità soddisfatta ATO")
            print("Differenza assoluta finale ATO:", abs(val_1 - val_2))
            break

        n_scenarios += 50
        iter_count += 1

    return n_scenarios, list_result

def inSampleStability_NewsVendor(n_scenarios_init, num_variables_NewsVendor, setting_NewsVendor, 
                                  parameters_NewsVendor, N, alpha, max_iter, tol):

    z_alpha = norm.ppf(1 - alpha / 2)
    n_scenarios = n_scenarios_init
    iter_count = 0

    list_result = []

    while iter_count < max_iter:
        phi_list = []

        for _ in range(N):
            stoch_model_1 = StochModel(num_variables_NewsVendor, setting_NewsVendor)
            stoch_model_2 = StochModel(num_variables_NewsVendor, setting_NewsVendor)

            initial_value = setting_NewsVendor.get('expectedValue')

            tree_1 = ScenarioTree("Tree NV", [n_scenarios], 1, initial_value, stoch_model_1)
            tree_2 = ScenarioTree("Tree NV", [n_scenarios], 1, initial_value, stoch_model_2)

            prob_1, demand_1 = stoch_model_1.simulate_one_time_step(None, n_scenarios)
            prob_2, demand_2 = stoch_model_2.simulate_one_time_step(None, n_scenarios)

            ottimo_1, val_1 = newsVendor(demand_1[0], prob_1, parameters_NewsVendor)
            ottimo_2, val_2 = newsVendor(demand_2[0], prob_2, parameters_NewsVendor)

            phi_list.append(val_1 - val_2)

        phi_array = np.array(phi_list)
        mu = np.mean(phi_array)
        sigma = np.std(phi_array, ddof=1)

        margin = z_alpha * sigma / np.sqrt(N)
        ci_lower = mu - margin
        ci_upper = mu + margin

        print(f"n_scenarios = {n_scenarios} | Intervallo di confidenza: [{ci_lower:.4f}, {ci_upper:.4f}]")

        list_result.append({
                'n_scenarios': n_scenarios,
                'mu': mu,
                'devstd' : sigma,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
        })

        # Condizione di arresto: 0 dentro l'intervallo di confidenza
        if ci_lower <= 0 <= ci_upper and abs(val_1 - val_2) < tol:
            print("Condizione di stabilità soddisfatta NV")
            print("Differenza assoluta finale NV:", abs(val_1 - val_2))
            break

        n_scenarios += 50
        iter_count += 1

    return n_scenarios, list_result











