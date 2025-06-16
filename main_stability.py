import numpy as np

from clustering.wasserstainDistance import *
from data.parameters import *
from problems.assembleToOrder import *
from problems.newsVendor import *
from result import *
from setting.scenarioSetting import *
from solvers import *
from stability.inSampleStability import *
from stability.outOfSampleStability import *
from result.results_ATO import *
from result.results_NV import *

from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist
from scipy.stats import norm


num_variables_ATO = parameters_ATO.get('n_items')
setting_ATO = scenario_setting_ATO

num_variables_NewsVendor = 1
setting_NewsVendor = scenario_setting_NewsVendor


"""In Sample Stability"""

n_scenarios = 10
np.random.seed(1)

print("########## IN-SAMPLE STABILITY ##########")

print("----ATO----")
n_scenarios_ATO = inSampleStability_ATO(n_scenarios, num_variables_ATO, setting_ATO, parameters_ATO, N=50, alpha=0.01, max_iter=100, tol = 10)
print('n_scenari_finale ATO', n_scenarios_ATO)

print("----NV----")
n_scenarios_NV = inSampleStability_NewsVendor(n_scenarios, num_variables_NewsVendor, setting_NewsVendor, parameters_NewsVendor, N=50, alpha=0.01, max_iter=100, tol=10)
print('n_scenari_finale NewsVendor', n_scenarios_NV)


"""Out of Sample Stability"""

print("########## OUT-OF-SAMPLE STABILITY ##########")

n_scenarios_train = 20
n_scenarios_test = 100
n_repeat = 10

print("----ATO----")
std_dev_ATO, cv_ATO = outOfSampleStability_ATO( n_scenarios_train, n_scenarios_test, num_variables_ATO, setting_ATO, n_repeat)
print('deviazione standard ATO', std_dev_ATO)
print('misura relativa di stabilità ATO', cv_ATO)

print("----NV----")
std_dev_NV, cv_NV = outOfSampleStability_NewsVendor( n_scenarios_train, n_scenarios_test, num_variables_NewsVendor, setting_NewsVendor, n_repeat)
print('deviazione standard NewsVendor', std_dev_NV)
print('misura relativa di stabilità NewsVendor', cv_NV)



# Salva risultati ATO
save_ato_results_stability(
    in_sample=n_scenarios_ATO,
    out_sample_std=std_dev_ATO,
    out_sample_cv=cv_ATO
)

# Salva risultati NewsVendor
save_newsvendor_results_stability(
    in_sample=n_scenarios_NV,
    out_sample_std=std_dev_NV,
    out_sample_cv=cv_NV
)


