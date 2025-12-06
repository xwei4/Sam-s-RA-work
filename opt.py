import gurobipy as gp
from gurobipy import GRB
import numpy as np

p_vals = np.linspace(0.01, 0.99, 10)
theta_vals = np.linspace(0.01, 0.99, 10)

grid = []

grid = []
for p in p_vals:
    for i, thetaL in enumerate(theta_vals[:-1]):
        for thetaH in theta_vals[i+1:]:
            grid.append((p, thetaL, thetaH))
print("Total grid points:", len(grid))

results = []

eps = 1e-4

for p, thetaL, thetaH in grid:
    m = gp.Model()
    m.setParam('OutputFlag', False)

    qH_star = m.addVar(lb=0.0, ub=1.0, name="qH*")
    qL_star = m.addVar(lb=0.0, ub=1.0, name="qL*")
    qHc = m.addVar(lb=0.0, ub=1.0, name="qHc")
    vqH_star = m.addVar(name="vqH*")
    vqL_star = m.addVar(name="vqL*")
    vqHc = m.addVar(name="vqHc")
    z = m.addVar()

    m.Params.NonConvex = 2

    m.addConstr(p*(vqL_star - thetaL*qL_star - (thetaH - thetaL)*qHc) + (1 - 
    p)*(vqHc - thetaH * qHc) == 1, name="c1")
    m.addConstr(qH_star - qHc >= eps,   "order1")
    m.addConstr(qL_star - qH_star >= eps, "order2")
    m.addConstr(z >= p*(vqL_star - thetaL*qL_star), name="c2")
    m.addConstr(z >= vqH_star - thetaH*qH_star, name="c3")
    m.addConstr(vqHc >= ((p*(thetaH - thetaL)+thetaH*(1-p))/(1-p))*qHc, name="c4")
    m.addConstr(vqH_star - vqHc <= ((p*(thetaH - thetaL)+thetaH*(1-p))/(1-p)) * qHc * (qH_star - qHc),name="c5")
    m.addConstr(vqH_star - vqHc >= thetaH * (qH_star - qHc),name="c6")
    m.addConstr(vqL_star - vqH_star <= thetaH * (qL_star - qH_star),name="c7")
    m.addConstr(vqL_star - vqH_star >= thetaL * (qL_star - qH_star),name="c8")
    m.setObjective(z,GRB.MINIMIZE)
    m.optimize()

    if m.status == GRB.OPTIMAL:
        obj = m.objVal
    else:
        obj = None

    results.append({
        "p": p,
        "thetaL": thetaL,
        "thetaH": thetaH,
        "obj": obj,
        "status": m.status,
    })

best = max(results, key=lambda r: r["obj"] if r["obj"] is not None else -1e100)
print("Best:", best, results)