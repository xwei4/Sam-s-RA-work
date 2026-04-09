import gurobipy as gp 
from gurobipy import Model, GRB
import numpy as np
import pandas as pd

eps = 1e-4

def solve_instance(p, thetaH, thetaL):
    """
    Builds and solves LP; returns (objective, feasible_flag)
    """
    try:
        m = build_and_solve_lp(p, thetaH, thetaL)

        if m.Status == GRB.OPTIMAL:
            return m.ObjVal, True
        else:
            return np.nan, False

    except Exception:
        return np.nan, False


def build_and_solve_lp(p, thetaH, thetaL):

    deltaTheta = thetaH - thetaL
   

    m = gp.Model("gerardi")
    m.Params.OutputFlag = 0  # silence solver output

    qHc     = m.addVar(lb=eps, name="qhc")
    qHStar  = m.addVar(lb=eps, name="qHstar")
    qLStar  = m.addVar(lb=eps, name="qLstar")
    VqHc    = m.addVar(lb=0, name="Vqhc")
    VqHStar = m.addVar(lb=0, name="VqHstar")
    VqLStar = m.addVar(lb=0, name="VqLStar")
    z       = m.addVar(lb=0, ub=1, name="dynamic_valuation")

    # Monotonicity
    m.addConstr(qHStar >= qHc + eps)
    m.addConstr(qLStar >= qHStar + eps)
    m.addConstr(VqHStar >= VqHc + eps)
    m.addConstr(VqLStar >= VqHStar + eps)

    # derivate of the value function at qHc
    S = p * deltaTheta / (1 - p) + thetaH

    # Linearized constraints
    m.addConstr(VqHc >= S * qHc)
    m.addConstr(VqHStar - VqHc <= S * (qHStar - qHc))
    m.addConstr(VqHStar - VqHc >= thetaH * (qHStar - qHc))
    m.addConstr(VqLStar - VqHStar <= thetaH * (qLStar - qHStar))
    m.addConstr(VqLStar - VqHStar >= thetaL * (qLStar - qHStar))

    # Individual Rationality: Utility (Payment - Cost) must be >= 0
    m.addConstr(VqLStar - thetaL * qLStar >= 0, name="IR_Low")
    m.addConstr(VqHStar - thetaH * qHStar >= 0, name="IR_High")
    m.addConstr(VqHc - thetaH * qHc >= 0, name="IR_Hc")

    # Incentive Compatibility_Low: Low type prefers L-contract over H-contract
    m.addConstr(VqLStar - thetaL * qLStar >= VqHStar - thetaL * qHStar, name="IC_Low")
    # IC_High: High type prefers H-contract over L-contract
    m.addConstr(VqHStar - thetaH * qHStar >= VqLStar - thetaH * qLStar, name="IC_High")

    m.addConstr(z >= thetaH * qHStar) # pooling
    m.addConstr(z >= p * thetaL * qLStar) # firing
    
    m.addConstr(p*(VqLStar - thetaL*qLStar - deltaTheta*qHc) + (1-p)*(VqHc - thetaH*qHc) == 1)

    m.setObjective(z, GRB.MINIMIZE)
    m.optimize()

    return m

def grid_search(p_values, thetaH_values, thetaL_values):
    results = []

    for p in p_values:
        for thetaH in thetaH_values:
            for thetaL in thetaL_values:

                if thetaL >= thetaH:
                    continue

                try:
                    obj, feasible = solve_instance(p, thetaH, thetaL)
                    results.append({
                        "p": p,
                        "thetaH": thetaH,
                        "thetaL": thetaL,
                        "objective": obj,
                        "feasible": feasible
                    })

                except Exception:
                    results.append({
                        "p": p,
                        "thetaH": thetaH,
                        "thetaL": thetaL,
                        "objective": np.nan,
                        "feasible": False
                    })

    return pd.DataFrame(results)


def find_best_overall(df):
    feasible_df = df[df["feasible"]]

    if feasible_df.empty:
        print("No feasible solutions in the grid.")
        return None

    return feasible_df.loc[feasible_df["objective"].idxmin()]


def main():
    # Define grids
    p_vals      = np.linspace(0.1, 0.99, 10)
    thetaL_vals = np.linspace(0.01, 10, 10)
    thetaH_vals = np.linspace(0.10, 10, 10)

    print("Running grid search...")
    df = grid_search(p_vals, thetaH_vals, thetaL_vals)

    # Print top solution
    best = find_best_overall(df)
    print("\nBest overall combination:\n", best)

    # Save results
    df.to_csv("grid_search_results.csv", index=False)
    print("\nSaved full results to grid_search_results.csv")

    return df, best

# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":
    df, best = main()