import numpy as np
import gurobipy as gp
from gurobipy import GRB


def solve_model(p, thetaL, thetaH, eps=1e-6):
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

    for r in range(rounds):

        print(f"\n  ROUND {r+1}  ")

        p_vals = np.linspace(
            max(0.01, p_center - p_width),
            min(0.95, p_center + p_width),
            7
        )

        thetaH_vals = np.linspace(
            max(0.5, thetaH_center - thetaH_width),
            thetaH_center + thetaH_width,
            8
        )

        ratio_vals = np.linspace(
            max(0.01, ratio_center - ratio_width),
            min(0.99, ratio_center + ratio_width),
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