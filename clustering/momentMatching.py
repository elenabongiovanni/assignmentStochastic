import numpy as np
import gurobipy as gp
from gurobipy import GRB

def momentmatching(mu,sigma2,weight,scenarios):
    
    len_scenarios = len(scenarios)
    scenarios = np.array(scenarios)
    model = gp.Model("momentmatching")
    model.setParam('OutputFlag', 0)

    # Variables
    pi = model.addVars(len_scenarios, lb=0, ub=1, name="prob")
    u = model.addVar(lb = 0, name = "first abs")
    v = model.addVar(lb = 0, name = "second abs")

    # Constraint
    model.addConstr(sum(pi[s] for s in range(len_scenarios)) == 1 , name = "somma prob")
    model.addConstr(u == sum(pi[s] * scenarios[s][i] for s in range(len_scenarios) for i in range(scenarios.shape[1])) -mu)
    #model.addConstr(u >= -sum(pi[s] * scenarios[s][i] for i in range(scenarios.shape[1])for s in range(len_scenarios))-mu)

    model.addConstr(v == sum(pi[s] *(scenarios[s][i]-mu)**2 for i in range(scenarios.shape[1]) for s in range(len_scenarios))-sigma2)
    #model.addConstr(v >= -gp.quicksum(pi[s] *(scenarios[s][i]-mu)^2 for i in range(scenarios.shape[1]) for s in range(len_scenarios))-sigma2)
    model.addConstr(u == gp.abs_(u))
    model.addConstr(v == gp.abs_(v))

    # Objective function
    #model.setObjective(abs(gp.quicksum(pi[s] * scenarios[s] for s in range(len_scenarios))-mu)+\
    #                    weight*abs(gp.quicksum(pi[s] *(scenarios[s]-mu)^2 for s in range(len_scenarios)))-sigma2, GRB.MINIMIZE)
    model.setObjective(u + weight*v, GRB.MINIMIZE)

    model.optimize()
    if model.status == GRB.OPTIMAL:
        prob = np.zeros(len_scenarios)
        for s in range(len_scenarios):
            prob[s] = pi[s].X
        momentmatching_distance = model.objVal
        return momentmatching_distance, prob
    else:
        raise Exception("Optimization problem did not converge!")


def reduce_scenarios_momentmatching(scenarios, num_reduce,mu,sigma2,weight):
    
    scenarios = np.array(scenarios)
    n = len(scenarios)
    # coppie generate da scenari random per clacolare delle distanze e ricavare una tolleranza    
    best_distance = 10000000
    best_subset = None

    for i in range(1,10):
        scenario_reduce = scenarios[np.random.choice(scenarios.shape[0],size = num_reduce, replace = False)]
        dist,prob = momentmatching(mu,sigma2,weight,scenarios)
        if dist < best_distance:
            best_distance = dist
            best_subset = scenario_reduce

    return best_subset, best_distance


