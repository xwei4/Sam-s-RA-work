import gurobipy as gp
from gurobipy import GRB
import numpy as np

p_vals = np.linspace(0.01, 0.99, 9)
theta_vals = np.linspace(0, 10, 10)

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

    qHc     = m.addVar(lb=eps, name="qhc")
    qHStar  = m.addVar(lb=eps, name="qHstar")
    qLStar  = m.addVar(lb=eps, name="qLstar")
    VqHc    = m.addVar(lb=0, name="Vqhc")
    VqHStar = m.addVar(lb=0, name="VqHstar")
    VqLStar = m.addVar(lb=0, name="VqLStar")
    z       = m.addVar(lb=0, ub=1, name="dynamic_valuation")

    m.Params.NonConvex = 2

    m.addConstr(p*(VqLStar - thetaL*qLStar - (thetaH - thetaL)*qHc) + (1 - p)*(VqHc - thetaH * qHc) == 1, name="sep")
    #m.addConstr(p*(vqL_star - thetaL*qL_star)== 1, name="firing")
    m.addConstr(qHStar - qHc >= eps, "c0")
    m.addConstr(qLStar - qHStar >= eps, "c1")
    m.addConstr(z >= p*(VqLStar - thetaL*qLStar), name="c2")
    m.addConstr(z >= VqHStar - thetaH*qHStar, name="c3")
    m.addConstr(VqHc >= ((p*(thetaH - thetaL)+thetaH*(1-p))/(1-p))*qHc, name="c4")
    m.addConstr(VqHStar - VqHc <= ((p*(thetaH - thetaL)+thetaH*(1-p))/(1-p)) * qHc * (qHStar - qHc),name="c5")
    m.addConstr(VqHStar - VqHc >= thetaH * (qHStar - qHc),name="c6")
    m.addConstr(VqLStar - VqHStar <= thetaH * (qLStar - qHStar),name="c7")
    m.addConstr(VqLStar - VqHStar >= thetaL * (qLStar - qHStar),name="c8")
    m.setObjective(z,GRB.MINIMIZE)
    m.optimize()

    if m.status == GRB.OPTIMAL:
        obj = m.objVal

        results.append({
            "p": p,
            "thetaL": thetaL,
            "thetaH": thetaH,
            "obj": obj,
            "status": m.status,
            "qH_star": qHStar.X,
            "qL_star": qLStar.X,
            "qHc": qHc.X,
            "vqH_star": VqHStar.X,
            "vqL_star": VqLStar.X,
            "vqHc": VqHc.X,
            "z": z.X,
        })
    else:
        results.append({
            "p": p,
            "thetaL": thetaL,
            "thetaH": thetaH,
            "obj": None,
            "status": m.status,
        })

        results.append({
            "p": p,
            "thetaL": thetaL,
            "thetaH": thetaH,
            "obj": obj,
            "status": m.status,
        })

best = max(results, key=lambda r: r["obj"] if r["obj"] is not None else -1e100)

print("Best result:")
for k, v in best.items():
    print(f"{k}: {v}")