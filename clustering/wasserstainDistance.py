import numpy as np
import gurobipy as gp
from gurobipy import GRB

def compute_cost_matrix_multivariate(points_mu, points_nu, p=2):

    points_mu = np.array(points_mu)
    points_nu = np.array(points_nu)

    m, d1 = points_mu.shape
    n, d2 = points_nu.shape

    if d1 != d2:
        raise ValueError("Points must have same dimension")

    cost_matrix = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            diff = np.abs(points_mu[i] - points_nu[j])
            cost_matrix[i, j] = np.linalg.norm(diff, ord=p)

    return cost_matrix

def wasserstein_distance(mu, nu, cost_matrix):
    
    m = len(mu)
    n = len(nu)

    # Model
    model = gp.Model("wasserstein")

    model.setParam("OutputFlag", 0)

    # Decision variables
    gamma = model.addVars(m, n, lb=0, ub=GRB.INFINITY, name="gamma")

    # Objective function
    model.setObjective(gp.quicksum(cost_matrix[i, j] * gamma[i, j] for i in range(m) for j in range(n)), GRB.MINIMIZE)

    # Constraints

    for i in range(m):
        model.addConstr(gp.quicksum(gamma[i, j] for j in range(n)) == mu[i], name=f"supply_{i}") # Total mass transported from each mu_i matches the corresponding mass in mu

    for j in range(n):
        model.addConstr(gp.quicksum(gamma[i, j] for i in range(m)) == nu[j], name=f"demand_{j}") # Total mass transported to each nu_j matches the corresponding mass in nu

    model.optimize()

    if model.status == GRB.OPTIMAL:
        transport_plan = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                transport_plan[i, j] = gamma[i, j].X
        wasserstein_distance = model.objVal
        return wasserstein_distance, transport_plan
    else:
        raise Exception("Optimization problem did not converge!")
    
def reduce_scenarios_wasserstein(scenarios, num_reduce, p=2):

    scenarios = np.array(scenarios)
    n = len(scenarios)
    mu = np.ones(n) / n  

    # Couples generated from random scenarios to evaluate distances and set a tolerance  
    best_distance = 10000000
    best_subset = None

    for i in range(1,10):
        scenario_reduce = scenarios[np.random.choice(scenarios.shape[0],size = num_reduce, replace = False)]
        nu = np.ones(num_reduce) / num_reduce
        cost = compute_cost_matrix_multivariate(scenarios,scenario_reduce,p=p)
        dist,plan = wasserstein_distance(mu,nu,cost)
        if dist < best_distance:
            best_distance = dist
            best_subset = scenario_reduce

    return best_subset, best_distance, nu
    