import numpy as np
import gurobipy as gp
from gurobipy import GRB


def solve_model(p, thetaL, thetaH, eps=1e-6):
        m = gp.Model()
        m.Params.OutputFlag = 0
        m.Params.NonConvex = 2
        m.Params.FeasibilityTol = 1e-8

        qH_star = m.addVar(lb=0.0)
        qL_star = m.addVar(lb=0.0)
        qHc = m.addVar(lb=0.0)

        vqH_star = m.addVar(lb=0.0)
        vqL_star = m.addVar(lb=0.0)
        vqHc = m.addVar(lb=0.0)

        z = m.addVar(lb=-1e6)

        slope = (p*(thetaH - thetaL) + thetaH*(1-p)) / (1-p)

        m.addConstr(p*(vqL_star - thetaL*qL_star - (thetaH-thetaL)*qHc)+ (1-p)*(vqHc - thetaH*qHc) == 1)
        #m.addConstr(p*(vqL_star - thetaL*qL_star)== 1)

        m.addConstr(qH_star - qHc >= eps)
        m.addConstr(qL_star - qH_star >= eps)

        m.addConstr(z >= p*(vqL_star - thetaL*qL_star))
        m.addConstr(z >= vqH_star - thetaH*qH_star)

        m.addConstr(vqHc >= slope*qHc)

        m.addConstr(vqH_star - vqHc <= slope*(qH_star - qHc))
        m.addConstr(vqH_star - vqHc >= thetaH*(qH_star - qHc))

        m.addConstr(vqL_star - vqH_star <= thetaH*(qL_star - qH_star))
        m.addConstr(vqL_star - vqH_star >= thetaL*(qL_star - qH_star))

        m.setObjective(z, GRB.MINIMIZE)

        m.optimize()

        if m.status == GRB.OPTIMAL:
            return {
                "p": p,
                "thetaL": thetaL,
                "thetaH": thetaH,
                "qH_star": qH_star.X,
                "qL_star": qL_star.X,
                "qHc": qHc.X,
                "vqH_star": vqH_star.X,
                "vqL_star": vqL_star.X,
                "vqHc": vqHc.X,
                "slope": slope,
                "objective_z": m.objVal
        }
        else:
            return None

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