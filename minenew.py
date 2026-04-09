import numpy as np
import gurobipy as gp
from gurobipy import GRB


def solve_model(p, thetaL, thetaH, eps=1e-6):
    deltaTheta = thetaH - thetaL

    m = gp.Model()
    m.Params.OutputFlag = 0

    m.Params.FeasibilityTol = 1e-9

    qHc = m.addVar(lb=eps, name="qhc")
    qHStar = m.addVar(lb=eps, name="qHstar")
    qLStar = m.addVar(lb=eps, name="qLstar")
    VqHc = m.addVar(lb=0, name="Vqhc")
    VqHStar = m.addVar(lb=0, name="VqHstar")
    VqLStar = m.addVar(lb=0, name="VqLStar")
    z = m.addVar(lb=0, ub=1, name="dynamic_valuation")

    m.addConstr(qHStar >= qHc + eps)
    m.addConstr(qLStar >= qHStar + eps)
    m.addConstr(VqHStar >= VqHc + eps)
    m.addConstr(VqLStar >= VqHStar + eps)

    S = p * deltaTheta / (1 - p) + thetaH

    m.addConstr(VqHc >= S * qHc)
    m.addConstr(VqHStar - VqHc <= S * (qHStar - qHc))
    m.addConstr(VqHStar - VqHc >= thetaH * (qHStar - qHc))
    m.addConstr(VqLStar - VqHStar <= thetaH * (qLStar - qHStar))
    m.addConstr(VqLStar - VqHStar >= thetaL * (qLStar - qHStar))

    m.addConstr(VqLStar - thetaL * qLStar >= 0, name="IR_Low")
    m.addConstr(VqHStar - thetaH * qHStar >= 0, name="IR_High")
    m.addConstr(VqHc - thetaH * qHc >= 0, name="IR_Hc")

    m.addConstr(VqLStar - thetaL * qLStar >= VqHStar - thetaL * qHStar, name="IC_Low")
    m.addConstr(VqHStar - thetaH * qHStar >= VqLStar - thetaH * qLStar, name="IC_High")

    m.addConstr(z >= VqHStar - thetaH * qHStar) 
    m.addConstr(z >= p * (VqLStar - thetaL * qLStar)) 

    m.addConstr(p*(VqLStar - thetaL*qLStar - deltaTheta*qHc) + (1-p)*(VqHc - thetaH*qHc) == 1) 

    m.setObjective(z, GRB.MINIMIZE)
    m.optimize()

    m.Params.NonConvex = 2

    if m.status != GRB.OPTIMAL:
        return None

    return {
        "p": p,
        "thetaL": thetaL,
        "thetaH": thetaH,
        "qHc": qHc.X,
        "qHStar": qHStar.X,
        "qLStar": qLStar.X,
        "VqHc": VqHc.X,
        "VqHStar": VqHStar.X,
        "VqLStar": VqLStar.X,
        "z": z.X,
        "objective_z": m.objVal,
    }

def search():

    p_center = 0.5
    thetaH_center = 5
    ratio_center = 0.5   # thetaL = ratio * thetaH

    p_width = 0.4
    thetaH_width = 4
    ratio_width = 0.4

    rounds = 3
    best = None

    eps=1e-6

    for r in range(rounds):

        print(f"\n  ROUND {r+1}  ")

        p_vals = np.linspace(
            max(eps, p_center - p_width),
            min(1-eps, p_center + p_width),
            7
        )

        thetaH_vals = np.linspace(
            max(0.5, thetaH_center - thetaH_width),
            thetaH_center + thetaH_width,
            8
        )

        ratio_vals = np.linspace(
            max(eps, ratio_center - ratio_width),
            min(1-eps, ratio_center + ratio_width),
            8
        )

        best_obj = float("inf")
        best = None

        for p in p_vals:
            for thetaH in thetaH_vals:
                for ratio in ratio_vals:

                    thetaL = ratio * thetaH

                    if thetaL >= thetaH:
                        continue

                    result = solve_model(p, thetaL, thetaH)

                    if result is None:
                        continue

                    if result["objective_z"] < best_obj:
                        best_obj = result["objective_z"]
                        best = result
        
        print("\nBest so far:")
        for k, v in best.items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")

        p_center = best["p"]
        thetaH_center = best["thetaH"]
        ratio_center = best["thetaL"] / best["thetaH"]

        p_width *= 0.4
        thetaH_width *= 0.4
        ratio_width *= 0.4

    return best



best_solution = search()

print("\n FINAL SOLUTION ")
for k, v in best_solution.items():
    if isinstance(v, float):
        print(f"{k}: {v:.6f}")
    else:
        print(f"{k}: {v}")