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


#### In Sample Stability ####

n_scenarios = 100
np.random.seed(1)

n_scenarios_ATO, list_results_ATO = inSampleStability_ATO(n_scenarios, num_variables_ATO, setting_ATO, parameters_ATO, N=100, alpha=0.01, max_iter=100, tol = 10)
n_scenarios_NV, list_results_NV = inSampleStability_NewsVendor(n_scenarios, num_variables_NewsVendor, setting_NewsVendor, parameters_NewsVendor, N=150, alpha=0.01, max_iter=100, tol=10)

#### Out of Sample Stability ####

n_scenarios_train = 20
n_scenarios_test = 100
n_repeat = 10

std_dev_ATO, cv_ATO = outOfSampleStability_ATO( n_scenarios_train, n_scenarios_test, num_variables_ATO, setting_ATO, n_repeat)
std_dev_NV, cv_NV = outOfSampleStability_NewsVendor( n_scenarios_train, n_scenarios_test, num_variables_NewsVendor, setting_NewsVendor, n_repeat)


#### Functions used to save the results ####

save_ato_results_stability(in_sample=n_scenarios_ATO, out_sample_std=std_dev_ATO, out_sample_cv=cv_ATO, history=list_results_ATO)
save_newsvendor_results_stability(in_sample=n_scenarios_NV, out_sample_std=std_dev_NV, out_sample_cv=cv_NV, history=list_results_NV)


