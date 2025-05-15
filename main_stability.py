import numpy as np

from clustering.wasserstainDistance import *
from data.parameters import *
from problems.assembleToOrder import *
from problems.newsVendor import *
from instances import *
from result import *
from setting.scenarioSetting import *
from solvers import *
from stability.inSampleStability import *
from stability.outOfSampleStability import *

from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist


num_variables_ATO = parameters_ATO.get('n_items')
setting_ATO = scenario_setting_ATO

num_variables_NewsVendor = 1
setting_NewsVendor = scenario_setting_NewsVendor


"""In Sample Stability"""

n_scenarios = 5
tol = 50

n_scenarios_ATO = inSampleStability_ATO(n_scenarios, num_variables_ATO, setting_ATO, tol)
print('n_scenari_finale ATO', n_scenarios_ATO)

n_scenarios_NV = inSampleStability_NewsVendor(n_scenarios, num_variables_NewsVendor, setting_NewsVendor, tol)
print('n_scenari_finale NewsVendor', n_scenarios_NV)


"""Out of Sample Stability"""

n_scenarios_train = 20
n_scenarios_test = 100
n_repeat = 10

std_dev_ATO, cv_ATO = outOfSampleStabilityATO( n_scenarios_train, n_scenarios_test, num_variables_ATO, setting_ATO, n_repeat)
print('deviazione standard ATO', std_dev_ATO)
print('misura relativa di stabilità ATO', cv_ATO)


std_dev_NV, cv_NV = outOfSampleStability_NewsVendor( n_scenarios_train, n_scenarios_test, num_variables_NewsVendor, setting_NewsVendor, n_repeat)
print('deviazione standard NewsVendor', std_dev_NV)
print('misura relativa di stabilità NewsVendor', cv_NV)



