import numpy as np
import gurobipy as gp
from gurobipy import GRB
import itertools

def compute_cost_matrix_multivariate(points_mu, points_nu, p=2):
    """
    Calcola la matrice dei costi tra punti multivariati usando la norma p.

    Parameters:
    -----------
    points_mu : array-like, shape (m, d)
        Scenari originali (m scenari, d dimensioni).
    points_nu : array-like, shape (n, d)
        Scenari ridotti (n scenari, d dimensioni).
    p : int, optional
        Norm p da usare (default = 2 → distanza euclidea).

    Returns:
    --------
    cost_matrix : ndarray, shape (m, n)
        Matrice dei costi: costo di trasportare massa da i a j.
    """
    points_mu = np.array(points_mu)
    points_nu = np.array(points_nu)

    # Estrae le dimensioni di points
    m, d1 = points_mu.shape
    n, d2 = points_nu.shape

    if d1 != d2:
        raise ValueError("Le due serie di punti devono avere la stessa dimensione")

    cost_matrix = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            diff = np.abs(points_mu[i] - points_nu[j])
            cost_matrix[i, j] = np.linalg.norm(diff, ord=p)

    return cost_matrix

def wasserstein_distance(mu, nu, cost_matrix):
    """
    Compute the 1-Wasserstein distance between two discrete distributions using Gurobi.

    Parameters:
    -----------
    mu : array-like, shape (m,)
        Probability distribution of the first set of points (source).
    nu : array-like, shape (n,)
        Probability distribution of the second set of points (target).
    cost_matrix : array-like, shape (m, n)
        The cost matrix where cost_matrix[i][j] is the cost of transporting mass from
        point i in mu to point j in nu.
    
    Returns:
    --------
    wasserstein_distance : float
        The computed Wasserstein distance between mu and nu.
    transport_plan : array, shape (m, n)
        The optimal transport plan.
    """
    m = len(mu)
    n = len(nu)

    # Create a Gurobi model
    model = gp.Model("wasserstein")

    # Disable Gurobi output (comment this if you want to see Gurobi's solver output)
    model.setParam("OutputFlag", 0)

    # Decision variables: transport plan gamma_ij
    gamma = model.addVars(m, n, lb=0, ub=GRB.INFINITY, name="gamma")

    # Objective: minimize the sum of the transport costs
    model.setObjective(gp.quicksum(cost_matrix[i, j] * gamma[i, j] for i in range(m) for j in range(n)), GRB.MINIMIZE)

    # Constraints: ensure that the total mass transported from each mu_i matches the corresponding mass in mu
    for i in range(m):
        model.addConstr(gp.quicksum(gamma[i, j] for j in range(n)) == mu[i], name=f"supply_{i}")

    # Constraints: ensure that the total mass transported to each nu_j matches the corresponding mass in nu
    for j in range(n):
        model.addConstr(gp.quicksum(gamma[i, j] for i in range(m)) == nu[j], name=f"demand_{j}")

    # Solve the optimization model
    model.optimize()

    # Extract the optimal transport plan and the Wasserstein distance
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
    """
    Riduce il numero di scenari multivariati usando la distanza di Wasserstein.
    """
    scenarios = np.array(scenarios)
    n = len(scenarios)
    mu = np.ones(n) / n  # probabilità uniformi

    combinations = list(itertools.combinations(range(n), num_reduce))
    best_distance = float("inf")
    best_subset = None

    for comb in combinations:
        reduced = scenarios[list(comb)]
        nu = np.ones(num_reduce) / num_reduce
        cost = compute_cost_matrix_multivariate(scenarios, reduced, p=p)
        dist, plan = wasserstein_distance(mu, nu, cost)
        if dist < best_distance:
            best_distance = dist
            best_subset = comb
        print(comb)

    return list(best_subset), scenarios[list(best_subset)], best_distance