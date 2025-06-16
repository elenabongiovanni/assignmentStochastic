import numpy as np
import gurobipy as gp
from gurobipy import GRB

def momentmatching(mu,sigma2,weight,scenarios):
    
    len_scenarios = len(scenarios)
    scenarios = np.array(scenarios)

    model = gp.Model("momentmatching")
    model.setParam('OutputFlag', 0)
    model.setParam("TimeLimit", 30)

    # Variables
    pi = model.addVars(len_scenarios, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="prob")
    
    #u = model.addVar(lb = 0, name = "first abs")
    #v = model.addVar(lb = 0, name = "second abs")

    # Constraint
    model.addConstr(gp.quicksum(pi[s] for s in range(len_scenarios)) == 1 , name = "somma prob")

    d = scenarios.shape[1]  # numero dimensioni (es. 3 se scenario è [x, y, z])

    mu = np.array(mu).flatten()
    sigma2 = np.array(sigma2).flatten()

    if mu.size == 1:
        mu = np.full(d, mu[0])  # copia la media su tutte le dimensioni

    if np.isscalar(sigma2):
        sigma2 = sigma2**2  # nel tuo caso devstd -> varianza
    sigma2 = np.array(sigma2).flatten()
    if sigma2.size == 1:
        sigma2 = np.full(d, sigma2[0])  # copia la varianza su tutte le dimensioni


    #mean = gp.quicksum(pi[s] * scenarios[s][i] for i in range(scenarios.shape[1]) for s in range(len_scenarios) )
    #var = gp.quicksum( pi[s]*(scenarios[s][i]-mu)**2 for i in range(scenarios.shape[1]) for s in range(len_scenarios))
   
    mean_expr = [gp.quicksum(pi[s] * scenarios[s][j] for s in range(len_scenarios)) for j in range(d)]
    var_expr = [gp.quicksum(pi[s] * (scenarios[s][j] - mu[j])**2 for s in range(len_scenarios)) for j in range(d)]

    mean_diff = gp.quicksum((mean_expr[j] - mu[j])**2 for j in range(d))
    var_diff = gp.quicksum((var_expr[j] - sigma2[j])**2 for j in range(d))

    #mean_diff = (mean - mu)**2
    #var_diff = (var - sigma2)**2

    #model.addConstr(u >= -sum(pi[s] * scenarios[s][i] for i in range(scenarios.shape[1])for s in range(len_scenarios))-mu)

    #model.addConstr(v == sum(pi[s] *(scenarios[s][i]-mu)**2 for i in range(scenarios.shape[1]) for s in range(len_scenarios))-sigma2)
    #model.addConstr(v >= -gp.quicksum(pi[s] *(scenarios[s][i]-mu)^2 for i in range(scenarios.shape[1]) for s in range(len_scenarios))-sigma2)
    #model.addConstr(u == gp.abs_(u))
    #model.addConstr(v == gp.abs_(v))

    # Objective function
    #model.setObjective(abs(gp.quicksum(pi[s] * scenarios[s] for s in range(len_scenarios))-mu)+\
    #                    weight*abs(gp.quicksum(pi[s] *(scenarios[s]-mu)^2 for s in range(len_scenarios)))-sigma2, GRB.MINIMIZE)
    
    objective = mean_diff + (var_diff * float(weight))
    model.setObjective(objective, GRB.MINIMIZE)

    # model.setObjective(mean_diff*mean_diff + var_diff*var_diff, GRB.MINIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        prob = np.zeros(len_scenarios)
        for s in range(len_scenarios):
            prob[s] = pi[s].X
        momentmatching_distance = model.objVal
        #print(momentmatching_distance)
        return momentmatching_distance, prob
    else:
        raise Exception("Optimization problem did not converge!")


def reduce_scenarios_momentmatching(scenarios, num_reduce,mu,sigma2,weight):

    n_item = scenarios.shape[1]
    scenarios = np.array(scenarios)
    n = len(scenarios)
    # coppie generate da scenari random per clacolare delle distanze e ricavare una tolleranza    
    best_distance = np.inf
    best_subset = None
    subset = np.zeros((n_item,num_reduce))

    for i in range(1,50):
        scenario_reduce = scenarios[np.random.choice(scenarios.shape[0],size = num_reduce, replace = False)]
        dist,prob = momentmatching(mu,sigma2,weight,scenario_reduce)
        if dist < best_distance:
            best_distance = dist
            best_subset = scenario_reduce
    
    for i in range(n_item):
        subset[i,:] = best_subset[:,i]

    return subset, best_distance, prob