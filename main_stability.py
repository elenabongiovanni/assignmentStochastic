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

from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist

n_scenarios = 5
tol = 50

num_variables_ATO = parameters_ATO.get('n_items')
setting_ATO = scenario_setting_ATO

num_variables_NewsVendor = 1
setting_NewsVendor = scenario_setting_NewsVendor

n_scenarios_ATO = inSaampleStability_ATO(n_scenarios, num_variables_ATO, setting_ATO, tol)
print('n_scenari_finale', n_scenarios_ATO)

#n_scenarios_NV = inSampleStability_NewsVendor(n_scenarios, num_variables_NewsVendor, setting_NewsVendor, tol)
#print('n_scenari_finale', n_scenarios_NV)